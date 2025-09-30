#!/usr/bin/env python3
"""
Enhanced JSON Parser for LLM Responses
处理各种复杂和兼容的JSON格式问题
"""

import os
import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJSONParser:
    """增强的JSON解析器，能够处理各种格式问题"""
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'error_types': {}
        }
    
    def clean_json_string(self, text: str) -> str:
        """清理JSON字符串，移除常见问题"""
        # 移除markdown代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # 移除行内注释 (// 注释)
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        
        # 移除数字后的括号注释，如 "3 (Neutral)" -> "3"
        text = re.sub(r'(\d+)\s*\([^)]*\)', r'\1', text)
        
        # 修复属性名没有双引号的问题
        text = re.sub(r'(\w+):', r'"\1":', text)
        
        # 移除多余的逗号
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        return text.strip()
    
    def extract_json_blocks(self, text: str) -> List[str]:
        """提取文本中所有可能的JSON块"""
        json_blocks = []
        
        # 方法1: 查找完整的JSON对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        json_blocks.extend(matches)
        
        # 方法2: 查找数组格式的JSON
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        array_matches = re.findall(array_pattern, text, re.DOTALL)
        json_blocks.extend(array_matches)
        
        # 方法3: 查找嵌套的JSON结构
        nested_pattern = r'\{.*?\}'
        nested_matches = re.findall(nested_pattern, text, re.DOTALL)
        json_blocks.extend(nested_matches)
        
        return json_blocks
    
    def try_parse_json(self, json_str: str) -> Optional[Dict]:
        """尝试解析JSON字符串"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def try_parse_with_ast(self, json_str: str) -> Optional[Dict]:
        """使用AST尝试解析类似Python字典的格式"""
        try:
            # 替换单引号为双引号
            json_str = json_str.replace("'", '"')
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return None
    
    def extract_scores_from_text(self, text: str) -> Dict[str, List[int]]:
        """从文本中提取分数信息"""
        result = {
            'ad_attitude_scores': [],
            'purchase_intention_scores': []
        }
        
        # 查找数字模式
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            nums = [int(n) for n in numbers if 1 <= int(n) <= 7]
            
            # 根据上下文判断分数类型
            if 'attitude' in text.lower() or 'like' in text.lower():
                result['ad_attitude_scores'] = nums[:4] if len(nums) >= 4 else nums
            if 'intention' in text.lower() or 'purchase' in text.lower():
                result['purchase_intention_scores'] = nums[4:7] if len(nums) >= 7 else nums[4:]
        
        return result
    
    def parse_response(self, response_text: str) -> Optional[Dict]:
        """解析LLM响应文本"""
        if not response_text.strip():
            return None
        
        # 清理文本
        cleaned_text = self.clean_json_string(response_text)
        
        # 方法1: 尝试直接解析清理后的文本
        parsed = self.try_parse_json(cleaned_text)
        if parsed:
            return parsed
        
        # 方法2: 提取JSON块并尝试解析
        json_blocks = self.extract_json_blocks(cleaned_text)
        for block in json_blocks:
            parsed = self.try_parse_json(block)
            if parsed:
                return parsed
        
        # 方法3: 尝试AST解析
        for block in json_blocks:
            parsed = self.try_parse_with_ast(block)
            if parsed:
                return parsed
        
        # 方法4: 从文本中提取分数信息
        scores = self.extract_scores_from_text(response_text)
        if scores['ad_attitude_scores'] or scores['purchase_intention_scores']:
            result = {'ad_type': 'unknown'}
            result.update(scores)
            return result
        
        return None
    
    def process_file(self, response_file: str, metadata_file: str) -> Tuple[bool, str, Optional[Dict]]:
        """处理单个响应文件"""
        try:
            # 读取响应文件
            with open(response_file, 'r', encoding='utf-8') as f:
                response_text = f.read()
            
            # 读取元数据文件
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # 解析响应
            parsed_json = self.parse_response(response_text)
            
            if parsed_json:
                # 合并元数据和解析结果
                result = {**metadata, **parsed_json}
                return True, "Success", result
            else:
                return False, "No valid JSON found", None
                
        except Exception as e:
            return False, str(e), None
    
    def process_experiment_directory(self, response_dir: str, json_output_dir: str):
        """处理整个实验目录"""
        logger.info(f"Processing experiment directory: {response_dir}")
        
        # 创建输出目录
        os.makedirs(json_output_dir, exist_ok=True)
        
        # 获取所有响应文件
        response_files = [f for f in os.listdir(response_dir) 
                         if f.endswith('.txt') and not f.endswith('_metadata.json')]
        
        success_count = 0
        error_count = 0
        error_details = []
        
        for response_file in response_files:
            self.stats['total_processed'] += 1
            
            response_path = os.path.join(response_dir, response_file)
            metadata_file = response_file.replace('.txt', '_metadata.json')
            metadata_path = os.path.join(response_dir, metadata_file)
            
            success, error_msg, result = self.process_file(response_path, metadata_path)
            
            if success:
                # 保存成功的JSON
                json_filename = response_file.replace('.txt', '.json')
                json_path = os.path.join(json_output_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                self.stats['successful'] += 1
            else:
                error_count += 1
                self.stats['failed'] += 1
                
                # 记录错误类型
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
                
                error_details.append((response_file, error_msg))
                
                # 保存错误信息
                error_filename = response_file.replace('.txt', '_parse_error.txt')
                error_path = os.path.join(json_output_dir, error_filename)
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Parse Error: {error_msg}\n\n")
                    f.write("Raw Output:\n")
                    # 重新读取响应文件内容
                    with open(response_path, 'r', encoding='utf-8') as rf:
                        f.write(rf.read())
        
        logger.info(f"Completed {response_dir}: {success_count} successful, {error_count} errors")
        
        # 显示错误详情
        if error_details:
            logger.info("Error details:")
            for file, error in error_details[:5]:  # 只显示前5个错误
                logger.info(f"  {file}: {error}")
            if len(error_details) > 5:
                logger.info(f"  ... and {len(error_details) - 5} more errors")
    
    def process_all_experiments(self, response_base_dir: str, json_base_dir: str):
        """处理所有实验"""
        logger.info("Starting enhanced JSON parsing...")
        
        if not os.path.exists(response_base_dir):
            logger.error(f"Response directory not found: {response_base_dir}")
            return
        
        # 遍历所有实验文件夹
        for experiment_dir in os.listdir(response_base_dir):
            experiment_path = os.path.join(response_base_dir, experiment_dir)
            if not os.path.isdir(experiment_path):
                continue
            
            # 创建对应的JSON输出目录
            json_experiment_dir = experiment_dir.replace('experiment_responses_', 'experiment_outputs_enhanced_')
            json_output_dir = os.path.join(json_base_dir, json_experiment_dir)
            
            self.process_experiment_directory(experiment_path, json_output_dir)
        
        # 打印统计信息
        self.print_statistics()
    
    def print_statistics(self):
        """打印统计信息"""
        logger.info("=== Enhanced JSON Parser Statistics ===")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
        
        logger.info("Error types:")
        for error_type, count in sorted(self.stats['error_types'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {error_type}: {count}")


def main():
    """主函数"""
    parser = EnhancedJSONParser()
    
    # 设置路径
    response_base_dir = "./response_output"
    json_base_dir = "./json_output_enhanced"
    
    # 处理所有实验
    parser.process_all_experiments(response_base_dir, json_base_dir)


if __name__ == "__main__":
    main()
