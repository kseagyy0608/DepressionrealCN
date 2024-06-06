import requests

def line_notify(message):
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": "Bearer cfipWYK4eqsKL1oOHh0pBXQaMEKH3EisQnBXDfVzuqU"
    }
    payload = {
        "message": message
    }
    response = requests.post(url, headers=headers, data=payload)
    return response

# if __name__ == "__main__":
#     token = "cfipWYK4eqsKL1oOHh0pBXQaMEKH3EisQnBXDfVzuqU"  # เปลี่ยนเป็น Token ที่คุณได้รับจาก Line Notify
#     message = "Hello, I'm ghosts"
#     response = line_notify("sumary_line")
#     if response.status_code == 200:
#         print("Message sent successfully!")
#     else:
#         print("Failed to send message. Status code:", response.status_code)
