"""Main Flask application."""
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import hmac
import secrets
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from backend.config import Config
from backend.services.colab_loader import get_colab_content
from backend.services.code_cleaner import clean_notebook_code
from backend.services.precheck import run_precheck
from backend.services.code_parser import parse_code
from backend.services.logger import get_logger
from backend.services.generator import generate_all_cases, generate_correct_alternative, generate_incorrect_case, _calculate_cost, _convert_usd_to_rub
from backend.services.evaluator import evaluate_homework
from backend.utils.syntax_validator import validate_python_syntax
from backend.models.prompts import PROMPTS, INCORRECT_CASE_TYPES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
file_logger = get_logger()

# Create Flask app
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
RUNS_FILE = Path(__file__).parent / 'data' / 'runs.json'
RUNS_LIMIT_SERVER = 50


def _load_runs_from_disk() -> list:
    """Load run history from disk, returning empty list on any error."""
    try:
        if RUNS_FILE.exists():
            return json.loads(RUNS_FILE.read_text(encoding='utf-8'))
    except Exception:
        pass
    return []


def _save_runs_to_disk(runs: list) -> None:
    """Persist run history to disk, keeping at most RUNS_LIMIT_SERVER entries."""
    RUNS_FILE.parent.mkdir(exist_ok=True)
    RUNS_FILE.write_text(
        json.dumps(runs[:RUNS_LIMIT_SERVER], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)
app.json.ensure_ascii = False  # For Russian characters

app.config['SECRET_KEY'] = Config.SECRET_KEY or secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = Config.SESSION_COOKIE_SECURE
app.config['SESSION_COOKIE_HTTPONLY'] = Config.SESSION_COOKIE_HTTPONLY
app.config['SESSION_COOKIE_SAMESITE'] = Config.SESSION_COOKIE_SAMESITE
app.config['PERMANENT_SESSION_LIFETIME'] = Config.PERMANENT_SESSION_LIFETIME

if not Config.SECRET_KEY:
    logger.warning('SECRET_KEY is not set. Generated ephemeral secret key for this process.')


def _is_authenticated() -> bool:
    """Check whether user has a valid authenticated session."""
    return bool(session.get('auth_ok') is True and session.get('auth_email') == Config.AUTH_EMAIL)


def _is_public_path(path: str) -> bool:
    """Allow public endpoints required before login."""
    if path in ('/', '/favicon.ico', '/api/health', '/api/auth/status', '/api/auth/login', '/api/auth/logout'):
        return True
    return path.startswith('/static/')


def _normalize_colab_code_for_execution(raw_code: str) -> str:
    """Prepare Colab export text into a stable code block for parsing/generation."""
    # Content with the cell-outputs marker was produced by cell-type-aware parsing
    # (external service or .ipynb extractor) — return as-is, no extra cleaning.
    if '# === CELL OUTPUTS ===' in raw_code:
        return raw_code
    return clean_notebook_code(raw_code)


def _format_curator_text(evaluation: dict | None, name: str, tema: str = '') -> str:
    """Format evaluation JSON into a human-readable curator message."""
    if not evaluation:
        return '—'

    parts = []

    # 1. Greeting
    parts.append(f'Добрый день, {name}!')

    # 2. Topic intro
    topic = tema.strip() if tema.strip() else (evaluation.get('topic') or '').strip()
    if topic:
        parts.append(f'\nОценил Вашу работу над домашним заданием по теме "{topic}"! Давайте рассмотрим его более детально:')

    # 3. Per-task breakdown
    homework_tasks = evaluation.get('homework_tasks') or []
    for idx, task in enumerate(homework_tasks):
        task_num = task.get('task_number', idx + 1)
        task_desc = (task.get('task_description') or '').strip()
        score = task.get('score', '')
        comment = (task.get('comment') or '').strip()
        task_lines = [f'\nЗадание {task_num}:']
        if task_desc:
            task_lines.append(f'Описание: {task_desc}')
        if score != '':
            task_lines.append(f'Оценка: {score}/10')
        if comment:
            task_lines.append(f'Комментарий: {comment}')
        parts.append('\n'.join(task_lines))

    # 4. Overall section
    overall_score = evaluation.get('overall_score', '')
    overall_comment = (evaluation.get('overall_comment') or '').strip()
    if overall_comment:
        overall_comment = re.sub(r'^\[АВТОМАТИЧЕСКАЯ ПРОВЕРКА\][^\n]*\n*', '', overall_comment).strip()
        # Strip greeting lines that the LLM may have embedded in the comment
        lines = overall_comment.split('\n')
        greeting_re = re.compile(
            r'(^(добрый|здравствуйте|привет)|' + re.escape(name) + r'.{0,30}!\s*$)',
            re.IGNORECASE
        )
        while lines and greeting_re.search(lines[0].strip()):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        overall_comment = '\n'.join(lines).strip()
    overall_lines = ['\nОбщая оценка задания:']
    if overall_score != '':
        overall_lines.append(f'Оценка: {overall_score}/10')
    if overall_comment:
        overall_lines.append(f'Общий комментарий: {overall_comment}')
    parts.append('\n'.join(overall_lines))

    # 5. Additional recommendations
    additional = (evaluation.get('additional_recommendations') or '').strip()
    if additional:
        parts.append(f'\nДополнительные рекомендации:\n{additional}')

    body = '\n'.join(parts)
    return f'{body}\n\nС уважением,\nКиберкуратор'


def _summarize_generation(results: list, etalon_code: str, trace: list) -> dict:
    """Build a compact, high-signal summary of a generation run.

    Designed to sit at the TOP of the trace log so it survives chat/preview
    truncation. Directly surfaces the «bug not applied → fell back to etalon»
    failure mode: any incorrect-type case whose code equals the etalon will be
    scored 10/10 by the evaluator, which is almost never intended.
    """
    INCORRECT_TYPES = {'syntax_error', 'logical_error', 'non_optimal', 'partial', 'cheating'}
    etalon_norm = (etalon_code or '').strip()

    per_case = []
    fallback_flags = []
    for i, r in enumerate(results):
        ctype = r.get('type', '')
        code = (r.get('student_code', '') or '').strip()
        identical = bool(code) and code == etalon_norm
        is_incorrect = ctype in INCORRECT_TYPES
        # An incorrect case identical to the etalon means the bug was not applied.
        suspicious = is_incorrect and identical
        if suspicious:
            fallback_flags.append(f"case[{i}] type={ctype} ИДЕНТИЧЕН эталону — баг НЕ применён → будет 10/10")
        per_case.append({
            'index': i,
            'type': ctype,
            'expected_grade': r.get('expected_grade'),
            'code_len': len(code),
            'identical_to_etalon': identical,
            'bug_not_applied': suspicious,
            'change_summary': (r.get('change_summary') or '')[:300],
        })

    # Pull only the validation / fallback verdicts from the trace — small and decisive.
    verdict_events = []
    for ev in (trace or []):
        name = ev.get('event', '')
        if name in (
            'incorrect_logical_validation', 'incorrect_non_optimal_validation',
            'incorrect_fallback', 'correct_fallback', 'incorrect_success',
        ):
            verdict_events.append({
                k: v for k, v in ev.items()
                if k in ('event', 'index', 'attempt', 'error_type', 'valid', 'reason')
            })

    return {
        'problems': fallback_flags or ['нет очевидных fallback-проблем (все incorrect-кейсы отличаются от эталона)'],
        'per_case': per_case,
        'validation_verdicts': verdict_events,
    }


def _format_evaluation_details(results: list) -> str:
    """Render a human-readable evaluation summary for each batch-check case."""
    lines = []
    for r in results:
        ev = r.get('evaluation') or {}
        precheck = ev.get('precheck') or {}
        actual = ev.get('overall_score', 'N/A')
        expected = r.get('expected_grade', 'N/A')
        match = "✓" if actual == expected else f"✗ (ожид. {expected})"
        lines.append(
            f"[{r['case_index']}] {r['type']} | оценка: {actual} {match}"
        )
        lines.append(
            f"    precheck → forced_score={precheck.get('forced_score')}, "
            f"syntax_error={precheck.get('has_syntax_errors')}, "
            f"stubs={precheck.get('has_stubs')}"
        )
        if precheck.get('reasons'):
            for reason in precheck['reasons']:
                lines.append(f"    ! {reason}")
        lines.append(f"    изменение: {(r.get('change_summary') or '')[:200]}")
        overall_comment = (ev.get('overall_comment') or '').replace('\n', ' ')
        lines.append(f"    общий комментарий: {overall_comment[:400]}")
        for task in ev.get('homework_tasks') or []:
            task_comment = (task.get('comment') or '').replace('\n', ' ')
            lines.append(
                f"    задание {task.get('task_number')}: score={task.get('score')} — "
                f"{task_comment[:300]}"
            )
        if r.get('error'):
            lines.append(f"    ОШИБКА: {r['error']}")
        lines.append("")
    return "\n".join(lines)

# Middleware: User identification
@app.before_request
def identify_user():
    """Apply auth guard and extract user identity for logs."""
    if not _is_public_path(request.path) and not _is_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401

    if _is_authenticated():
        request.user_email = session.get('auth_email', Config.AUTH_EMAIL)
        return None

    email = (
        request.headers.get('x-applet-user-email') or
        request.headers.get('x-applet-email') or
        request.headers.get('x-goog-authenticated-user-email') or
        'unknown@example.com'
    )
    request.user_email = email.replace('accounts.google.com:', '')


@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Return current auth state for frontend bootstrap."""
    return jsonify({
        'authenticated': _is_authenticated(),
        'email': session.get('auth_email') if _is_authenticated() else None,
    })


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Login with fixed email/password from env and issue secure session cookie."""
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    password = str(data.get('password', '')).strip()

    if not Config.AUTH_PASSWORD:
        return jsonify({'error': 'Server auth password is not configured'}), 500

    is_email_ok = hmac.compare_digest(email, Config.AUTH_EMAIL)
    is_password_ok = hmac.compare_digest(password, Config.AUTH_PASSWORD)

    if not (is_email_ok and is_password_ok):
        return jsonify({'error': 'Invalid credentials'}), 401

    session.clear()
    session.permanent = True
    session['auth_ok'] = True
    session['auth_email'] = Config.AUTH_EMAIL
    return jsonify({'ok': True, 'email': Config.AUTH_EMAIL})


@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    """Logout and clear session cookie."""
    session.clear()
    return jsonify({'ok': True})


# === FRONTEND ===
@app.route('/')
def index():
    """Serve frontend dev tool."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


# === RUN HISTORY ===
@app.route('/api/runs', methods=['GET'])
def api_get_runs():
    """Return stored run history."""
    return jsonify(_load_runs_from_disk())


@app.route('/api/runs', methods=['PUT'])
def api_put_runs():
    """Replace stored run history with the supplied list."""
    runs = request.get_json(silent=True)
    if not isinstance(runs, list):
        return jsonify({'error': 'Expected list'}), 400
    _save_runs_to_disk(runs)
    return jsonify({'ok': True})


# === HEALTH CHECK ===
@app.route('/api/health', methods=['GET'])
def health():
    """Check application and service health."""
    return jsonify({
        'status': 'ok',
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'openai_key_present': bool(Config.OPENAI_API_KEY),
        'log_dir': Config.LOG_DIR,
        'rubles_in_dollar': Config.RUBLES_IN_DOLLAR,
    })


# === FETCH COLAB ===
@app.route('/api/fetch-colab', methods=['POST'])
def fetch_colab():
    """
    Fetch and parse Colab content.
    
    Request body:
    {
        "url": "https://colab.research.google.com/drive/..."
    }
    
    Response:
    {
        "content": "clean python code",
        "raw_code": "original code",
        "precheck": {...},
        "parsed": {...}
    }
    """
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        assignment_type = data.get('assignment_type', 'code')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        logger.info(f"[{request.user_email}] Fetching Colab: {url}, assignment_type={assignment_type}")
        
        # Fetch content
        raw_code = get_colab_content(url, assignment_type=assignment_type)
        normalized_code = _normalize_colab_code_for_execution(raw_code)
        
        # Run precheck
        precheck = run_precheck(raw_code)
        precheck_dict = {
            'has_valid_code': precheck.has_valid_code,
            'has_syntax_errors': precheck.has_syntax_errors,
            'has_stubs': precheck.has_stubs,
            'has_metadata': precheck.has_metadata,
            'forced_score': precheck.forced_score,
            'reasons': precheck.reasons
        }
        
        # Parse code
        parsed = parse_code(normalized_code)
        parsed_dict = {
            'executable_code': parsed.executable_code,
            'logs': parsed.logs,
            'metadata': parsed.metadata
        }
        
        # Log operation
        file_logger.write_detailed_log('fetch_colab_success', {
            'url': url,
            'user': request.user_email,
            'raw_code_length': len(raw_code),
            'cleaned_code_length': len(normalized_code),
            'precheck': precheck_dict,
            'parsed': parsed_dict
        })
        
        file_logger.log_usage(request.user_email, 0, 'Fetch Colab')
        
        return jsonify({
            'content': normalized_code,
            'raw_code': raw_code,
            'precheck': precheck_dict,
            'parsed': parsed_dict
        })
    
    except Exception as e:
        logger.error(f"[{request.user_email}] Fetch Colab error: {str(e)}")
        
        file_logger.write_detailed_log('fetch_colab_error', {
            'url': request.get_json().get('url', 'unknown') if request.get_json() else 'unknown',
            'user': request.user_email,
            'error': str(e)
        })
        
        return jsonify({'error': str(e)}), 500


# === GET LOGS ===
@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent usage logs."""
    try:
        log_file = os.path.join(Config.LOG_DIR, 'usage_logs.json')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                import json
                logs = json.load(f)
                return jsonify(logs)
        return jsonify([])
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return jsonify([])


@app.route('/api/detailed-logs', methods=['GET'])
def list_detailed_logs():
    """List all detailed operation logs."""
    try:
        files = os.listdir(Config.LOG_DIR)
        log_files = [
            f for f in files
            if f.endswith('.txt')
        ]
        log_files.sort(reverse=True)
        return jsonify(log_files)
    except Exception as e:
        logger.error(f"Error listing logs: {str(e)}")
        return jsonify([])


@app.route('/api/detailed-logs/<filename>', methods=['GET'])
def get_detailed_log(filename):
    """Get content of a detailed log file."""
    try:
        # Security: prevent directory traversal
        if '..' in filename or '/' in filename:
            return jsonify({'error': 'Access denied'}), 403
        
        log_file = os.path.join(Config.LOG_DIR, filename)
        if not os.path.exists(log_file):
            return jsonify({'error': 'Log file not found'}), 404
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}")
        return jsonify({'error': str(e)}), 500


# === GENERATE CASES ===
@app.route('/api/generate-cases', methods=['POST'])
def generate_cases():
    """
    Generate test cases based on etalon Colab link.

    Request body:
    {
        "etalon_link": "https://...",
        "tema": "Тема задания",
        "zadanie": "Текст задания",
        "num_correct": 2,
        "num_incorrect": 3,
        "model": "gpt-4o",            // optional
        "enabled_types": ["syntax_error", "logical_error", "non_optimal"]  // optional
    }

    Response:
    {
        "results": [
            {
                "type": "perfect_etalon",
                "student_code": "...",
                "expected_grade": 10,
                "change_summary": "Эталонное решение"
            },
            ...
        ],
        "etalon_code": "...",
        "total_tokens": 1234
    }
    """
    generation_id = uuid.uuid4().hex
    # Progressive pipeline snapshot — populated stage by stage so that even on
    # failure the error log shows how far loading/parsing got and with what content.
    colab_pipeline = {'stage': 'init'}
    try:
        data = request.get_json()
        etalon_link = data.get('etalon_link', '').strip()
        tema = data.get('tema', '').strip()
        zadanie = data.get('zadanie', '').strip()
        num_correct = int(data.get('num_correct', 0))
        num_incorrect = int(data.get('num_incorrect', 0))
        model = data.get('model', 'gpt-4o')
        enabled_types = data.get('enabled_types')  # Optional list

        if not etalon_link:
            return jsonify({'error': 'etalon_link is required', 'generation_id': generation_id}), 400
        if not tema or not zadanie:
            return jsonify({'error': 'tema and zadanie are required', 'generation_id': generation_id}), 400
        if num_correct + num_incorrect == 0:
            return jsonify({'error': 'Specify num_correct or num_incorrect > 0', 'generation_id': generation_id}), 400

        # Get API key (from header or env)
        api_key = request.headers.get('x-openai-key') or None

        logger.info(
            f"[{request.user_email}] generate-cases: "
            f"correct={num_correct}, incorrect={num_incorrect}, model={model}"
        )

        # Load and prepare etalon code for generation.
        # Some Colab exports contain markdown/report prose mixed with code; this breaks syntax checks.
        # Each stage is recorded into colab_pipeline so the trace log (and the
        # error log on failure) shows exactly what the loader returned, how it
        # was normalized, and how parse_code split it.
        colab_pipeline['stage'] = 'loading_colab'
        raw_etalon = get_colab_content(etalon_link)
        colab_pipeline['colab_raw_content'] = raw_etalon
        colab_pipeline['colab_raw_length'] = len(raw_etalon)

        colab_pipeline['stage'] = 'normalizing'
        cleaned_etalon = _normalize_colab_code_for_execution(raw_etalon)
        colab_pipeline['normalized_etalon'] = cleaned_etalon
        colab_pipeline['normalized_etalon_length'] = len(cleaned_etalon)

        colab_pipeline['stage'] = 'parsing'
        parsed_etalon = parse_code(cleaned_etalon)
        colab_pipeline['parsed_executable_code'] = parsed_etalon.executable_code
        colab_pipeline['parsed_executable_length'] = len(parsed_etalon.executable_code)
        colab_pipeline['parsed_logs'] = parsed_etalon.logs
        colab_pipeline['parsed_logs_length'] = len(parsed_etalon.logs or '')
        colab_pipeline['parsed_metadata'] = parsed_etalon.metadata

        etalon_code = parsed_etalon.executable_code.strip() or cleaned_etalon
        colab_pipeline['used_parsed_executable'] = bool(parsed_etalon.executable_code.strip())

        colab_pipeline['stage'] = 'validating_syntax'
        etalon_syntax = validate_python_syntax(etalon_code)
        colab_pipeline['etalon_syntax_valid'] = etalon_syntax.get('valid', False)
        colab_pipeline['etalon_syntax_error'] = etalon_syntax.get('error')

        colab_pipeline['stage'] = 'generating'

        # Generate all cases
        result = generate_all_cases(
            etalon_code=etalon_code,
            tema=tema,
            zadanie=zadanie,
            num_correct=num_correct,
            num_incorrect=num_incorrect,
            model=model,
            enabled_types=enabled_types,
            api_key=api_key,
            generation_id=generation_id,
        )

        # Main operation log
        file_logger.write_detailed_log('generate_cases', {
            'user': request.user_email,
            'generation_id': generation_id,
            'etalon_link': etalon_link,
            'tema': tema,
            'num_correct': num_correct,
            'num_incorrect': num_incorrect,
            'model': model,
            'total_cases': len(result['results']),
            'total_tokens': result['total_tokens'],
        })

        # Dedicated full debug log for this generation request.
        file_logger.write_named_log(
            f'generation_{generation_id}.txt',
            'generate_cases_full_trace',
            {
                'generation_id': generation_id,
                'user': request.user_email,
                'request': {
                    'etalon_link': etalon_link,
                    'tema': tema,
                    'zadanie': zadanie,
                    'num_correct': num_correct,
                    'num_incorrect': num_incorrect,
                    'model': model,
                    'enabled_types': enabled_types,
                },
                'decision_summary': _summarize_generation(
                    result['results'], etalon_code, result.get('debug_log', [])
                ),
                'colab_pipeline': colab_pipeline,
                'etalon_code': etalon_code,
                'etalon_code_length': len(etalon_code),
                'etalon_syntax_valid': etalon_syntax.get('valid', False),
                'etalon_syntax_error': etalon_syntax.get('error'),
                'result_summary': {
                    'total_cases': len(result['results']),
                    'total_tokens': result['total_tokens'],
                    'total_prompt_tokens': result.get('total_prompt_tokens', 0),
                    'total_completion_tokens': result.get('total_completion_tokens', 0),
                    'total_cost_usd': result.get('total_cost_usd', 0.0),
                },
                'results': result['results'],
                'trace': result.get('debug_log', []),
            }
        )
        file_logger.log_usage(
            request.user_email,
            result['total_tokens'],
            f'Generate Cases: {num_correct}C + {num_incorrect}I'
        )

        return jsonify({
            'results': result['results'],
            'etalon_code': etalon_code,
            'total_tokens': result['total_tokens'],
            'total_cost_usd': result.get('total_cost_usd', 0.0),
            'total_cost_rub': _convert_usd_to_rub(result.get('total_cost_usd', 0.0)),
            'generation_id': generation_id,
        })

    except Exception as e:
        logger.error(f"[{request.user_email}] generate-cases error: {str(e)}")
        file_logger.write_named_log(
            f'generation_{generation_id}.txt',
            'generate_cases_error',
            {
                'generation_id': generation_id,
                'user': request.user_email,
                'failed_at_stage': colab_pipeline.get('stage'),
                'error': str(e),
                'colab_pipeline': colab_pipeline,
            }
        )
        file_logger.write_detailed_log('generate_cases_error', {
            'user': request.user_email,
            'generation_id': generation_id,
            'error': str(e),
        })
        return jsonify({'error': str(e), 'generation_id': generation_id}), 500


@app.route('/api/regenerate-case', methods=['POST'])
def regenerate_case():
    """
    Regenerate a single case with a custom description.

    Request body:
    {
        "etalon_link": "https://...",
        "tema": "...",
        "zadanie": "...",
        "case_type": "logical_error",
        "custom_desc": "...",
        "model": "gpt-4o"
    }
    """
    try:
        data = request.get_json()
        etalon_link = data.get('etalon_link', '').strip()
        tema = data.get('tema', '').strip()
        zadanie = data.get('zadanie', '').strip()
        case_type = data.get('case_type', '').strip()
        custom_desc = data.get('custom_desc', '').strip()
        model = data.get('model', 'gpt-4o')

        if not etalon_link or not tema or not zadanie or not case_type:
            return jsonify({'error': 'etalon_link, tema, zadanie, case_type are required'}), 400

        api_key = request.headers.get('x-openai-key') or None

        raw_etalon = get_colab_content(etalon_link)
        cleaned_etalon = _normalize_colab_code_for_execution(raw_etalon)
        parsed_etalon = parse_code(cleaned_etalon)
        etalon_code = parsed_etalon.executable_code.strip() or cleaned_etalon

        if case_type == 'perfect_alternative':
            case, usage = generate_correct_alternative(
                etalon_code=etalon_code,
                tema=tema,
                zadanie=zadanie,
                index=1,
                model=model,
                api_key=api_key,
                custom_hint=custom_desc or None,
            )
        else:
            case, usage = generate_incorrect_case(
                etalon_code=etalon_code,
                tema=tema,
                zadanie=zadanie,
                error_type=case_type,
                index=1,
                model=model,
                api_key=api_key,
                override_desc=custom_desc or None,
            )

        total_tokens = usage['prompt_tokens'] + usage['completion_tokens']
        total_cost_usd = _calculate_cost(model, usage['prompt_tokens'], usage['completion_tokens'])
        file_logger.log_usage(request.user_email, total_tokens, f'Regenerate Case: {case_type}')

        return jsonify({
            'type': case.type,
            'student_code': case.student_code,
            'expected_grade': case.expected_grade,
            'change_summary': case.change_summary,
            'total_tokens': total_tokens,
            'total_cost_usd': total_cost_usd,
            'total_cost_rub': _convert_usd_to_rub(total_cost_usd),
        })

    except Exception as e:
        logger.error(f"[{request.user_email}] regenerate-case error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generation-logs', methods=['GET'])
def list_generation_logs():
    """List dedicated generation trace logs."""
    try:
        files = os.listdir(Config.LOG_DIR)
        gen_logs = [
            f for f in files
            if f.startswith('generation_') and f.endswith('.txt')
        ]
        gen_logs.sort(reverse=True)
        return jsonify(gen_logs)
    except Exception as e:
        logger.error(f"Error listing generation logs: {str(e)}")
        return jsonify([])


@app.route('/api/generation-logs/<generation_id>', methods=['GET'])
def get_generation_log(generation_id):
    """Get dedicated generation trace log by generation ID."""
    try:
        if not re.match(r'^[a-zA-Z0-9_-]{6,128}$', generation_id or ''):
            return jsonify({'error': 'Invalid generation_id'}), 400

        filename = f'generation_{generation_id}.txt'
        log_file = os.path.join(Config.LOG_DIR, filename)
        if not os.path.exists(log_file):
            return jsonify({'error': 'Generation log not found'}), 404

        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error reading generation log: {str(e)}")
        return jsonify({'error': str(e)}), 500


# === CHECK HOMEWORK ===
@app.route('/api/check-homework', methods=['POST'])
def check_homework():
    """
    Evaluate a single student submission via LLM.

    Request body:
    {
        "tema": "Тема",
        "zadanie": "Текст задания",
        "resheniye": "Код студента",
        "name": "Иван",          // optional
        "model": "gpt-4o"        // optional
    }

    Response: full evaluation JSON + usage stats
    """
    try:
        data = request.get_json()
        tema = data.get('tema', '').strip()
        zadanie = data.get('zadanie', '').strip()
        resheniye = data.get('resheniye', '').strip()
        name = data.get('name', 'Участник')
        model = data.get('model', 'gpt-4o')
        execution_logs = (data.get('execution_logs') or '').strip()
        assignment_type = data.get('assignment_type', 'code')

        if not resheniye:
            return jsonify({'error': 'resheniye is required'}), 400
        if not zadanie:
            return jsonify({'error': 'zadanie is required'}), 400

        api_key = request.headers.get('x-openai-key') or None

        logger.info(f"[{request.user_email}] check-homework: model={model}, assignment_type={assignment_type}")

        result = evaluate_homework(
            tema=tema,
            zadanie=zadanie,
            resheniye=resheniye,
            name=name,
            model=model,
            api_key=api_key,
            execution_logs=execution_logs,
            assignment_type=assignment_type,
        )

        # Log operation
        file_logger.write_detailed_log('check_homework', {
            'user': request.user_email,
            'tema': tema,
            'zadanie': zadanie,
            'name': name,
            'model': model,
            'resheniye_len': len(resheniye),
            'resheniye_preview': resheniye[:1200],
            'execution_logs_len': len(execution_logs),
            'execution_logs_preview': execution_logs[:1200],
            'overall_score': result.get('overall_score'),
            'forced_score': result.get('precheck', {}).get('forced_score'),
            'usage': result.get('usage'),
            'result': result,
        })
        file_logger.log_usage(
            request.user_email,
            result.get('usage', {}).get('total_tokens', 0),
            'Check Homework'
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"[{request.user_email}] check-homework error: {str(e)}")
        file_logger.write_detailed_log('check_homework_error', {
            'user': request.user_email,
            'error': str(e),
        })
        return jsonify({'error': str(e)}), 500


# === RUN BATCH EVALUATION ===
@app.route('/api/run-batch-check', methods=['POST'])
def run_batch_check():
    """
    Evaluate all generated cases against the assignment.

    Request body:
    {
        "cases": [
            {"type": "...", "student_code": "...", "expected_grade": 10, "change_summary": "..."},
            ...
        ],
        "tema": "Тема",
        "zadanie": "Текст задания",
        "model": "gpt-4o"   // optional
    }

    Response:
    {
        "results": [
            {
                "case_index": 0,
                "type": "perfect_etalon",
                "expected_grade": 10,
                "change_summary": "...",
                "evaluation": { ...full evaluation result... },
                "error": null
            },
            ...
        ],
        "total_tokens": 5678
    }
    """
    try:
        data = request.get_json()
        cases = data.get('cases', [])
        tema = data.get('tema', '').strip()
        zadanie = data.get('zadanie', '').strip()
        model = data.get('model', 'gpt-4o')
        execution_logs = (data.get('execution_logs') or '').strip()
        assignment_type = data.get('assignment_type', 'code')

        if not cases:
            return jsonify({'error': 'cases list is required'}), 400
        if not zadanie:
            return jsonify({'error': 'zadanie is required'}), 400

        api_key = request.headers.get('x-openai-key') or None
        logger.info(
            f"[{request.user_email}] batch-check: {len(cases)} cases, model={model}, assignment_type={assignment_type}"
        )

        # non_optimal is valid working code — evaluated like perfect cases:
        # same etalon execution_logs + post-validator enabled.
        PERFECT_TYPES = {"perfect_etalon", "perfect_alternative", "non_optimal"}

        # Pre-pass: resolve execution_logs and cache_key for every case
        case_metas = []
        for i, case in enumerate(cases):
            case_type = case.get("type", "")
            student_code = case.get("student_code", "") or ""
            per_case_logs = case.get('execution_logs')
            if per_case_logs is not None:
                case_execution_logs = per_case_logs
            elif case_type in PERFECT_TYPES:
                case_execution_logs = execution_logs
            else:
                case_execution_logs = ""
            cache_input = "\n||\n".join([model, tema, zadanie, assignment_type, case_execution_logs, student_code])
            cache_key = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
            case_metas.append({
                'i': i, 'case': case, 'case_type': case_type,
                'student_code': student_code,
                'case_execution_logs': case_execution_logs,
                'cache_key': cache_key,
            })

        # Deduplicate: only evaluate each unique cache_key once
        seen_keys: set = set()
        to_eval = []
        for meta in case_metas:
            if meta['cache_key'] not in seen_keys:
                seen_keys.add(meta['cache_key'])
                to_eval.append(meta)

        # Evaluate unique cases in parallel
        def _do_eval(meta):
            try:
                ev = evaluate_homework(
                    tema=tema,
                    zadanie=zadanie,
                    resheniye=meta['student_code'],
                    name=Config.STUDENT_NAME,
                    model=model,
                    api_key=api_key,
                    execution_logs=meta['case_execution_logs'],
                    enable_post_validator=(meta['case_type'] in PERFECT_TYPES),
                    assignment_type=assignment_type,
                )
                return meta['cache_key'], ev, None
            except Exception as exc:
                return meta['cache_key'], None, str(exc)

        key_to_ev: dict = {}
        if to_eval:
            with ThreadPoolExecutor(max_workers=min(8, len(to_eval))) as executor:
                futures = [executor.submit(_do_eval, m) for m in to_eval]
                for future in as_completed(futures):
                    ck, ev, err = future.result()
                    key_to_ev[ck] = {'evaluation': ev, 'error': err}

        # Build ordered results
        results = []
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        eval_cache: dict = {}

        for meta in case_metas:
            i = meta['i']
            case = meta['case']
            ck = meta['cache_key']
            is_cached = ck in eval_cache

            if is_cached:
                cached = eval_cache[ck]
                evaluation = cached['evaluation']
                case_result = {
                    "case_index": i,
                    "type": case.get("type"),
                    "expected_grade": case.get("expected_grade"),
                    "change_summary": case.get("change_summary"),
                    "evaluation": evaluation,
                    "curator_text": _format_curator_text(evaluation, Config.STUDENT_NAME, tema),
                    "error": None,
                    "cost_usd": 0.0,
                    "cost_rub": 0.0,
                    "cached": True,
                    "cached_from_case_index": cached["case_index"],
                }
                logger.info(
                    f"  case {i}: type={case.get('type')}, reused cached evaluation "
                    f"from case {cached['case_index']} (score={evaluation.get('overall_score') if evaluation else None})"
                )
            else:
                ev_data = key_to_ev.get(ck, {})
                evaluation = ev_data.get('evaluation')
                err = ev_data.get('error')
                case_result = {
                    "case_index": i,
                    "type": case.get("type"),
                    "expected_grade": case.get("expected_grade"),
                    "change_summary": case.get("change_summary"),
                    "evaluation": evaluation,
                    "curator_text": _format_curator_text(evaluation, Config.STUDENT_NAME, tema) if evaluation else (err or '—'),
                    "error": err,
                    "cost_usd": 0.0,
                    "cost_rub": 0.0,
                    "cached": False,
                }
                if evaluation:
                    usage = evaluation.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    case_result["cost_usd"] = _calculate_cost(model, prompt_tokens, completion_tokens)
                    case_result["cost_rub"] = _convert_usd_to_rub(case_result["cost_usd"])
                    eval_cache[ck] = {"case_index": i, "evaluation": evaluation}
                if err:
                    logger.error(f"  case {i}: error — {err}")

            score = case_result["evaluation"].get("overall_score") if case_result["evaluation"] else None
            logger.info(
                f"  case {i}: type={case.get('type')}, score={score} (expected={case.get('expected_grade')})"
            )
            results.append(case_result)


        total_cost_usd = _calculate_cost(model, total_prompt_tokens, total_completion_tokens)

        file_logger.write_detailed_log('batch_check', {
            'user': request.user_email,
            'tema': tema,
            'zadanie': zadanie,
            'num_cases': len(cases),
            'execution_logs_len': len(execution_logs),
            'execution_logs_preview': execution_logs[:2000],
            'total_tokens': total_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_cost_usd': total_cost_usd,
            'total_cost_rub': _convert_usd_to_rub(total_cost_usd),
            'model': model,
            'cases_request': [
                {
                    'index': i,
                    'type': c.get('type'),
                    'expected_grade': c.get('expected_grade'),
                    'change_summary': c.get('change_summary'),
                    'student_code_len': len(c.get('student_code', '')),
                    'student_code_preview': (c.get('student_code', '') or '')[:1200],
                }
                for i, c in enumerate(cases)
            ],
            'summary': [
                {
                    "index": r["case_index"],
                    "type": r["type"],
                    "expected": r["expected_grade"],
                    "actual": r["evaluation"].get("overall_score") if r["evaluation"] else None,
                    "error": r["error"],
                    "cost_usd": r.get("cost_usd", 0.0),
                    "cost_rub": r.get("cost_rub", 0.0),
                }
                for r in results
            ],
            'evaluation_details': _format_evaluation_details(results),
            'results': results,
        })
        file_logger.log_usage(
            request.user_email,
            total_tokens,
            f'Batch Check ({len(cases)} cases)'
        )

        return jsonify({
            'results': results,
            'total_tokens': total_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_cost_usd': total_cost_usd,
            'total_cost_rub': _convert_usd_to_rub(total_cost_usd),
        })

    except Exception as e:
        logger.error(f"[{request.user_email}] batch-check error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# === PROMPTS ===
@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Return all system prompts for display."""
    return jsonify({
        'prompts': {
            'evaluation_system': {
                'title': 'Система проверки домашних заданий',
                'description': 'Системный промпт для LLM-проверяющей — оценивает качество решения',
                'content': PROMPTS.get('evaluation_system', ''),
                'type': 'system',
            },
            'evaluation_user': {
                'title': 'Пользовательский запрос проверки',
                'description': 'Пользовательский промпт с задачей и решением для проверки',
                'content': PROMPTS.get('evaluation_user', ''),
                'type': 'user',
            },
            'generation_correct_system': {
                'title': 'Генерация корректных вариантов',
                'description': 'Системный промпт для создания альтернативных правильных решений',
                'content': PROMPTS.get('generation_correct_system', ''),
                'type': 'system',
            },
            'generation_incorrect_system': {
                'title': 'Генерация ошибочных кейсов',
                'description': 'Системный промпт для создания синтетических ошибок в коде',
                'content': PROMPTS.get('generation_incorrect_system', ''),
                'type': 'system',
            },
        },
        'error_types': [
            {
                'type': item.get('type', ''),
                'name': item.get('type_name', ''),
                'description': item.get('description', ''),
            }
            for item in INCORRECT_CASE_TYPES
        ]
    })


# === ERROR HANDLERS ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
