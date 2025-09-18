"""
LLM Conversation Logger for MAT_HPO_LIB
Logs all LLM interactions for analysis and debugging
Adapted from MAT_HPO_LLM implementation
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch


class LLMConversationLogger:
    """
    Comprehensive logger for LLM conversations and decisions
    
    Records all LLM interactions, parsing attempts, decision logic,
    and performance outcomes for detailed analysis.
    """
    
    def __init__(self, dataset_name: str, mode: str, output_dir: str = None):
        """
        Initialize conversation logger
        
        Args:
            dataset_name: Name of dataset being optimized
            mode: Mode string (e.g., "alpha_0.3", "adaptive_0.01")
            output_dir: Output directory for logs
        """
        self.dataset_name = dataset_name
        self.mode = mode
        
        if output_dir is None:
            output_dir = f"./llm_logs/{dataset_name}"
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.conversations_file = os.path.join(
            output_dir, f"llm_conversations_{mode}_{timestamp}.jsonl")
        self.decisions_file = os.path.join(
            output_dir, f"llm_decisions_{mode}_{timestamp}.jsonl")
        self.summary_file = os.path.join(
            output_dir, f"llm_summary_{mode}_{timestamp}.json")
        
        # Statistics
        self.total_llm_calls = 0
        self.successful_parses = 0
        self.failed_parses = 0
        self.total_attempts = 0
        
        print(f"📝 LLM Conversation Logger initialized")
        print(f"   Dataset: {dataset_name}")
        print(f"   Mode: {mode}")
        print(f"   Output dir: {output_dir}")
    
    def log_llm_conversation(self, 
                           step: int,
                           attempt: int,
                           prompt: str,
                           response: str,
                           parse_success: bool,
                           parsed_params: Optional[Dict] = None,
                           error_message: Optional[str] = None,
                           dataset_info: Optional[Dict] = None,
                           training_history: Optional[List] = None):
        """Log a single LLM conversation attempt"""
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'attempt': attempt,
            'dataset_name': self.dataset_name,
            'mode': self.mode,
            'prompt': prompt,
            'response': response,
            'parse_success': parse_success,
            'parsed_params': parsed_params,
            'error_message': error_message,
            'dataset_info': dataset_info,
            'training_history_length': len(training_history) if training_history else 0
        }
        
        # Write to conversations log
        self._append_to_jsonl(self.conversations_file, conversation_entry)
        
        # Update statistics
        self.total_llm_calls += 1
        self.total_attempts += 1
        if parse_success:
            self.successful_parses += 1
        else:
            self.failed_parses += 1
    
    def log_final_decision(self,
                         step: int,
                         decision_source: str,
                         rl_actions: torch.Tensor,
                         final_actions: torch.Tensor,
                         executed_hyperparams: List[float],
                         performance: Dict[str, float],
                         llm_attempts: int = 0,
                         llm_success: bool = True,
                         fallback_used: bool = False):
        """Log final decision and its outcome"""
        
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'dataset_name': self.dataset_name,
            'mode': self.mode,
            'decision_source': decision_source,
            'rl_actions': rl_actions.tolist() if isinstance(rl_actions, torch.Tensor) else [],
            'final_actions': final_actions.tolist() if isinstance(final_actions, torch.Tensor) else [],
            'executed_hyperparams': executed_hyperparams,
            'performance': performance,
            'llm_attempts': llm_attempts,
            'llm_success': llm_success,
            'fallback_used': fallback_used
        }
        
        # Write to decisions log
        self._append_to_jsonl(self.decisions_file, decision_entry)
    
    def log_failure_summary(self,
                          step: int,
                          all_attempts: List[Dict],
                          fallback_params: Dict):
        """Log comprehensive failure analysis"""
        
        failure_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'dataset_name': self.dataset_name,
            'mode': self.mode,
            'failure_type': 'all_llm_attempts_failed',
            'total_attempts': len(all_attempts),
            'attempt_details': all_attempts,
            'fallback_params': fallback_params
        }
        
        # Write to conversations log
        self._append_to_jsonl(self.conversations_file, failure_entry)
        
        # Update statistics
        self.failed_parses += len(all_attempts)
        self.total_attempts += len(all_attempts)
    
    def _append_to_jsonl(self, filename: str, data: Dict):
        """Append data to JSONL file"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ Failed to write to {filename}: {e}")
    
    def save_final_summary(self, optimization_stats: Dict):
        """Save final summary statistics"""
        
        summary = {
            'dataset_name': self.dataset_name,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'llm_statistics': {
                'total_llm_calls': self.total_llm_calls,
                'total_attempts': self.total_attempts,
                'successful_parses': self.successful_parses,
                'failed_parses': self.failed_parses,
                'success_rate': self.successful_parses / max(self.total_attempts, 1),
                'avg_attempts_per_call': self.total_attempts / max(self.total_llm_calls, 1)
            },
            'optimization_statistics': optimization_stats,
            'log_files': {
                'conversations': os.path.basename(self.conversations_file),
                'decisions': os.path.basename(self.decisions_file),
                'summary': os.path.basename(self.summary_file)
            }
        }
        
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📊 Final summary saved to {self.summary_file}")
        except Exception as e:
            print(f"❌ Failed to save summary: {e}")
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'total_llm_calls': self.total_llm_calls,
            'total_attempts': self.total_attempts,
            'successful_parses': self.successful_parses,
            'failed_parses': self.failed_parses,
            'success_rate': self.successful_parses / max(self.total_attempts, 1) if self.total_attempts > 0 else 0,
            'avg_attempts_per_call': self.total_attempts / max(self.total_llm_calls, 1) if self.total_llm_calls > 0 else 0
        }
    
    def print_statistics(self):
        """Print current statistics"""
        stats = self.get_statistics()
        print(f"\n📊 LLM Conversation Statistics:")
        print(f"   Total LLM calls: {stats['total_llm_calls']}")
        print(f"   Total attempts: {stats['total_attempts']}")
        print(f"   Successful parses: {stats['successful_parses']}")
        print(f"   Failed parses: {stats['failed_parses']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Avg attempts per call: {stats['avg_attempts_per_call']:.1f}")


class SimpleLogger:
    """Simple logger for basic LLM logging when full logger is not needed"""
    
    def __init__(self):
        self.entries = []
    
    def log(self, message: str, data: Dict = None):
        """Log a simple message"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'data': data or {}
        }
        self.entries.append(entry)
        print(f"📝 {message}")
    
    def get_logs(self) -> List[Dict]:
        """Get all logged entries"""
        return self.entries.copy()
    
    def clear(self):
        """Clear all logged entries"""
        self.entries.clear()