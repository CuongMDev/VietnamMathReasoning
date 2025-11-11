#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool để lọc dữ liệu JSON sử dụng Gemini API
Nhập prompt và tập dữ liệu JSON, tool sẽ sử dụng LLM để lọc ra các dữ liệu thỏa mãn yêu cầu
"""

import json
import argparse
import sys
from typing import List, Dict, Any
import google.generativeai as genai
from pathlib import Path


def load_json_data(input_source: str) -> List[Dict[Any, Any]]:
    """
    Load dữ liệu JSON từ file hoặc từ string
    
    Args:
        input_source: Đường dẫn file JSON hoặc chuỗi JSON
    
    Returns:
        List các dictionary chứa dữ liệu
    """
    # Kiểm tra nếu là đường dẫn file
    if Path(input_source).exists():
        with open(input_source, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Thử parse như JSON string
        try:
            data = json.loads(input_source)
        except json.JSONDecodeError:
            raise ValueError(f"Không thể parse JSON từ: {input_source}")
    
    # Đảm bảo data là list
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError("Dữ liệu JSON phải là object hoặc array")
    
    return data


def filter_with_gemini(data: List[Dict[Any, Any]], prompt: str, api_key: str, model_name: str = "gemini-2.5-flash") -> List[Dict[Any, Any]]:
    """
    Sử dụng Gemini API để lọc dữ liệu theo prompt
    
    Args:
        data: Danh sách các dictionary cần lọc
        prompt: Prompt mô tả điều kiện lọc
        api_key: API key cho Gemini
        model_name: Tên model Gemini sử dụng (mặc định: gemini-pro)
    
    Returns:
        Danh sách các dictionary thỏa mãn điều kiện
    """
    # Cấu hình Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Chuyển đổi dữ liệu sang JSON string để gửi cho LLM
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    
    # Tạo prompt đầy đủ cho LLM
    full_prompt = f"""Bạn là một công cụ lọc dữ liệu chuyên nghiệp. 
Nhiệm vụ của bạn là lọc ra các bản ghi từ tập dữ liệu JSON sau đây thỏa mãn điều kiện được mô tả.

Điều kiện lọc: {prompt}

Tập dữ liệu JSON:
{data_json}

Yêu cầu:
1. Phân tích từng bản ghi trong tập dữ liệu
2. Chỉ giữ lại các bản ghi thỏa mãn điều kiện lọc
3. Trả về kết quả dưới dạng JSON array, chỉ chứa các bản ghi được lọc
4. Không thêm bất kỳ giải thích hay comment nào, chỉ trả về JSON thuần túy
5. Giữ nguyên cấu trúc dữ liệu gốc của mỗi bản ghi

Trả về chỉ JSON array, không có text nào khác:"""

    try:
        # Gọi Gemini API
        response = model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        # Loại bỏ markdown code blocks nếu có
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse kết quả
        filtered_data = json.loads(response_text)
        
        # Đảm bảo kết quả là list
        if isinstance(filtered_data, dict):
            filtered_data = [filtered_data]
        
        return filtered_data
    
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {str(e)}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Tool lọc dữ liệu JSON sử dụng Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Từ file JSON
  python filter_data.py -p "lọc các sản phẩm có giá > 100" -f data.json -k YOUR_API_KEY
  
  # Từ stdin (JSON string)
  echo '[{"name": "A", "price": 50}, {"name": "B", "price": 150}]' | python filter_data.py -p "lọc sản phẩm có giá > 100" -k YOUR_API_KEY
  
  # Lưu kết quả vào file
  python filter_data.py -p "lọc người dùng có tuổi > 18" -f users.json -k YOUR_API_KEY -o filtered_users.json
        """
    )
    
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Prompt mô tả điều kiện lọc dữ liệu"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Đường dẫn file JSON chứa dữ liệu (nếu không có sẽ đọc từ stdin)"
    )
    
    parser.add_argument(
        "-k", "--api-key",
        required=True,
        help="API key cho Gemini API"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="gemini-2.5-flash",
        help="Tên model Gemini sử dụng (mặc định: gemini-2.5-flash)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Đường dẫn file để lưu kết quả (nếu không có sẽ in ra stdout)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="In kết quả JSON với định dạng đẹp (pretty print)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load dữ liệu
        if args.file:
            data = load_json_data(args.file)
        else:
            # Đọc từ stdin
            stdin_data = sys.stdin.read()
            if not stdin_data.strip():
                print("Lỗi: Không có dữ liệu đầu vào. Vui lòng cung cấp file hoặc dữ liệu qua stdin.", file=sys.stderr)
                sys.exit(1)
            data = load_json_data(stdin_data)
        
        print(f"Đã load {len(data)} bản ghi từ dữ liệu đầu vào.", file=sys.stderr)
        print(f"Đang lọc dữ liệu với prompt: {args.prompt}", file=sys.stderr)
        
        # Lọc dữ liệu
        filtered_data = filter_with_gemini(data, args.prompt, args.api_key, args.model)
        
        print(f"Đã lọc được {len(filtered_data)} bản ghi thỏa mãn điều kiện.", file=sys.stderr)
        
        # Xuất kết quả
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2 if args.pretty else None)
            print(f"Đã lưu kết quả vào {args.output}", file=sys.stderr)
        else:
            if args.pretty:
                print(json.dumps(filtered_data, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(filtered_data, ensure_ascii=False))
    
    except KeyboardInterrupt:
        print("\nĐã hủy bởi người dùng.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

