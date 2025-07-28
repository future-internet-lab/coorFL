import argparse

# Khởi tạo parser
parser = argparse.ArgumentParser(description="Client khởi động tham số")

# Đối số --client_id: số nguyên, bắt buộc
parser.add_argument('--client_id', type=int, required=True, help='ID của client (số nguyên)')

# Đối số --device: chuỗi, không bắt buộc, mặc định là "cpu"
parser.add_argument('--device', type=str, default='cpu', help='Thiết bị sử dụng (ví dụ: cpu, cuda, cuda:0)')

# Phân tích đối số
args = parser.parse_args()

# In ra để kiểm tra
print(f'Client ID: {args.client_id}')
print(f'Device: {args.device}')
