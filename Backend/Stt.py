# import requests
# import os
# import json
# from datetime import datetime


# # API configuration
# API_BASE_URL = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
# API_KEY ='pbkGo03t_QKnDGrRFL85F3dC0w32cVCt2hm7ysnxy2'

# headers = {
#     'accept': 'application/json',
#     'Authorization': f'Bearer {API_KEY}'
# }

# file_path = r"C:\Users\SBA\Downloads\Hindi.mp3"

# with open(file_path, "rb") as file:
#     files = {
#         "file": (file_path, file)  
#     }
#     data = {
#         "language_code": "en-IN"  
#     }

#     response = requests.post(API_BASE_URL, headers=headers, files=files, data=data)
#     print(response.status_code)
#     print(response.text)

# result = response.json()
# print(result)

# # Save result to JSON file
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_filename = f"stt_result_{timestamp}.json"

# with open(output_filename, 'w', encoding='utf-8') as json_file:
#     json.dump(result, json_file, indent=4, ensure_ascii=False)

# print(f"\nResult saved to: {output_filename}")




# import requests
# import os
# import json
# from datetime import datetime

# # API configuration
# API_BASE_URL = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
# API_KEY = 'pbkGo03t_QKnDGrRFL85F3dC0w32cVCt2hm7ysnxy2'

# headers = {
#     'accept': 'application/json',
#     'Authorization': f'Bearer {API_KEY}'
# }

# file_path = r"C:\Users\SBA\Downloads\Hindi.mp3"

# try:
#     with open(file_path, "rb") as file:
#         files = {
#             "file": (os.path.basename(file_path), file)
#         }
#         data = {
#             "language_code": "en-IN"
#         }

#         response = requests.post(API_BASE_URL, headers=headers, files=files, data=data, timeout=60)

#     print("Status Code:", response.status_code)

#     # Check if response is JSON
#     if "application/json" in response.headers.get("Content-Type", ""):
#         result = response.json()
#     else:
#         print("⚠️ Non-JSON response received:")
#         print(response.text)
#         result = {"error": "Non-JSON response", "status_code": response.status_code, "body": response.text}

# except requests.exceptions.Timeout:
#     print("⏳ Request timed out.")
#     result = {"error": "Request timed out"}

# except requests.exceptions.RequestException as e:
#     print("❌ Request failed:", str(e))
#     result = {"error": str(e)}

# # Save result to JSON file
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_filename = f"stt_result_{timestamp}.json"

# with open(output_filename, 'w', encoding='utf-8') as json_file:
#     json.dump(result, json_file, indent=4, ensure_ascii=False)

# print(f"\n✅ Result saved to: {output_filename}")



# import requests

# url = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
# headers = {
#     'accept': 'application/json',
#     'Authorization': 'Bearer pbkGo03t_QKnDGrRFL85F3dC0w32cVCt2hm7ysnxy2'
# }

# file_path = r"C:\Users\SBA\Downloads\Hindi.mp3"
# with open(file_path, "rb") as file:
#     files = {
#         "file": (file_path, file)  # let requests guess MIME type
#     }
#     data = {
#         "language_code": "en-IN"  # or "hi-IN"
#     }

#     response = requests.post(url, headers=headers, files=files, data=data)

# print(response.status_code)
# print(response.json())



from sarvamai import SarvamAI

client = SarvamAI(
    api_subscription_key="sk_4x0i67xw_cteCF8N83pNl8l0K9gfhc7lQ",
)

# Option 1: Use raw string (prefix with 'r')
response = client.speech_to_text.transcribe(
    file=open(r"C:\Users\SBA\Downloads\GST\Licencing ( AHT - 10 MTS)\TLY - 28Jul2025 - 00083.WAV", "rb"),
    model="saarika:v2.5",
    language_code="en-IN"
)

print(response)





