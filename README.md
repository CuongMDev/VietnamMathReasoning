# Tool Lọc Dữ Liệu JSON với Gemini API

Tool này cho phép bạn lọc dữ liệu JSON bằng cách sử dụng Gemini API (Google's LLM) để hiểu và áp dụng các điều kiện lọc phức tạp thông qua prompt tự nhiên.

## Cài đặt

1. Cài đặt Python 3.7 trở lên
2. Cài đặt các dependencies:

```bash
pip install -r requirements.txt
```

3. Lấy API key từ Google AI Studio: https://makersuite.google.com/app/apikey

## Sử dụng

### Cú pháp cơ bản

```bash
python filter_data.py -p "PROMPT" -f INPUT_FILE.json -k YOUR_API_KEY
```

### Các tham số

- `-p, --prompt`: Prompt mô tả điều kiện lọc (bắt buộc)
- `-f, --file`: Đường dẫn file JSON chứa dữ liệu (tùy chọn, nếu không có sẽ đọc từ stdin)
- `-k, --api-key`: API key cho Gemini API (bắt buộc)
- `-m, --model`: Tên model Gemini sử dụng (mặc định: `gemini-pro`)
- `-o, --output`: Đường dẫn file để lưu kết quả (tùy chọn, nếu không có sẽ in ra stdout)
- `--pretty`: In kết quả JSON với định dạng đẹp

## Ví dụ sử dụng

### Ví dụ 1: Lọc từ file JSON

Tạo file `products.json`:
```json
[
  {"name": "Laptop", "price": 1500, "category": "Electronics"},
  {"name": "Book", "price": 20, "category": "Education"},
  {"name": "Phone", "price": 800, "category": "Electronics"},
  {"name": "Tablet", "price": 400, "category": "Electronics"}
]
```

Lọc các sản phẩm có giá > 500:
```bash
python filter_data.py -p "lọc các sản phẩm có giá lớn hơn 500" -f products.json -k YOUR_API_KEY --pretty
```

Kết quả:
```json
[
  {
    "name": "Laptop",
    "price": 1500,
    "category": "Electronics"
  },
  {
    "name": "Phone",
    "price": 800,
    "category": "Electronics"
  }
]
```

### Ví dụ 2: Lọc từ stdin

```bash
echo '[{"name": "John", "age": 25}, {"name": "Jane", "age": 17}]' | python filter_data.py -p "lọc người có tuổi >= 18" -k YOUR_API_KEY
```

### Ví dụ 3: Lưu kết quả vào file

```bash
python filter_data.py -p "lọc các bản ghi có trạng thái là 'active'" -f data.json -k YOUR_API_KEY -o filtered_data.json
```

### Ví dụ 4: Lọc với điều kiện phức tạp

```bash
python filter_data.py -p "lọc các đơn hàng có tổng giá trị > 1000 và trạng thái là 'pending'" -f orders.json -k YOUR_API_KEY --pretty
```

## Lưu ý

1. **API Key**: Bạn cần có API key từ Google AI Studio. API key sẽ được sử dụng để gọi Gemini API.

2. **Định dạng dữ liệu**: Dữ liệu đầu vào phải là JSON hợp lệ, có thể là:
   - Một object (sẽ được chuyển thành array có 1 phần tử)
   - Một array các objects

3. **Prompt**: Viết prompt bằng tiếng Việt hoặc tiếng Anh để mô tả điều kiện lọc. Gemini API sẽ hiểu và áp dụng điều kiện đó.

4. **Chi phí**: Sử dụng Gemini API có thể phát sinh chi phí tùy theo gói dịch vụ bạn đăng ký.

5. **Hiệu suất**: Với tập dữ liệu lớn, quá trình lọc có thể mất thời gian. Cân nhắc chia nhỏ dữ liệu nếu cần.

## Xử lý lỗi

- Nếu file JSON không tồn tại hoặc không hợp lệ, tool sẽ báo lỗi
- Nếu API key không hợp lệ, tool sẽ báo lỗi từ Gemini API
- Nếu prompt không rõ ràng, kết quả có thể không chính xác

## Giấy phép

Tool này được cung cấp miễn phí để sử dụng.

