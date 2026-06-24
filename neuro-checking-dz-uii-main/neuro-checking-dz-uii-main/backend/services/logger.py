"""Logging service - handles all operation logging."""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any
from backend.config import Config


class FileLogger:
    """Detailed file logger for all operations."""
    
    def __init__(self):
        self.log_dir = Config.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _render_output(self, operation_name: str, data: Dict[str, Any]) -> str:
        """Render dictionary payload as a readable text log."""
        output = f"--- DETAILED LOG: {operation_name} ---\n"
        output += f"Timestamp: {datetime.now().isoformat()}\n"
        output += "---" * 20 + "\n\n"

        for key, value in data.items():
            output += f"=== {key.upper()} ===\n"
            if isinstance(value, str):
                output += value
            elif value is None:
                output += "None"
            else:
                try:
                    output += json.dumps(value, indent=2, ensure_ascii=False)
                except:
                    output += str(value)
            output += "\n\n"

        output += "---" * 20 + "\n"
        output += "END OF LOG\n"
        return output

    def write_named_log(self, filename: str, operation_name: str, data: Dict[str, Any]) -> str:
        """
        Write detailed log using a predefined file name.

        Args:
            filename: File name (must end with .txt)
            operation_name: Name of the operation
            data: Dictionary with operation data

        Returns:
            Path to written log file
        """
        safe_name = os.path.basename(filename)
        if not safe_name.endswith('.txt'):
            safe_name += '.txt'

        filepath = os.path.join(self.log_dir, safe_name)
        output = self._render_output(operation_name, data)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)

        self.logger.info(f"Named detailed log written: {safe_name}")
        return filepath
    
    def write_detailed_log(self, operation_name: str, data: Dict[str, Any]) -> str:
        """
        Write detailed operation log to text file.
        
        Args:
            operation_name: Name of the operation (e.g., 'fetch_colab', 'generate_cases')
            data: Dictionary with operation data
            
        Returns:
            Path to written log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp}_{operation_name}.txt"
        filepath = os.path.join(self.log_dir, filename)
        output = self._render_output(operation_name, data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        
        self.logger.info(f"Detailed log written: {filename}")
        return filepath
    
    def log_usage(self, email: str, tokens: int, action: str) -> Dict[str, Any]:
        """
        Log API usage for billing purposes.
        
        Args:
            email: User email
            tokens: Number of tokens used
            action: Action description
            
        Returns:
            Log entry dict
        """
        cost_rub = (tokens * Config.RUB_PER_TOKEN)
        
        log_entry = {
            "email": email or "unknown@example.com",
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens,
            "cost_rub": round(cost_rub, 4),
            "action": action
        }
        
        # Append to JSON log file
        log_file = os.path.join(self.log_dir, "usage_logs.json")
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"Usage logged: {email} - {action} ({tokens} tokens, {log_entry['cost_rub']} RUB)"
        )
        
        return log_entry


# Global logger instance
_logger_instance = None


def get_logger() -> FileLogger:
    """Get or create logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FileLogger()
    return _logger_instance
