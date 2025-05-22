import json
import requests

# Send a GET request to the API root
r = requests.get("http://127.0.0.1:8000/")
print(f"GET Status Code: {r.status_code}")
print(f"GET Response: {r.json()['message']}")

# Define POST payload
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Send a POST request
r = requests.post("http://127.0.0.1:8000/data/", json=data)
print(f"POST Status Code: {r.status_code}")
print(f"POST Result: {r.json()['result']}")
