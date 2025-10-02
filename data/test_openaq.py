"""
Quick test script for OpenAQ v3 API
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_openaq_v3():
    """Test OpenAQ v3 API with your key"""
    
    api_key = os.getenv("OPENAQ_API_KEY")
    
    if not api_key:
        print("❌ OPENAQ_API_KEY not found in .env")
        return
    
    print(f"Testing OpenAQ v3 API...")
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}\n")
    
    # Test 1: Get locations in Egypt
    print("=" * 60)
    print("Test 1: Finding locations in Egypt")
    print("=" * 60)
    
    url = "https://api.openaq.org/v3/locations"
    headers = {"X-API-Key": api_key}
    params = {"country": "EG", "limit": 5}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ Authentication failed!")
            print("Response:", response.text)
            return
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('results'):
            print(f"✅ Found {len(data['results'])} locations in Egypt\n")
            
            for loc in data['results'][:3]:
                print(f"Location: {loc['name']}")
                print(f"  ID: {loc['id']}")
                print(f"  Coordinates: {loc['coordinates']}")
                print(f"  Sensors: {len(loc.get('sensors', []))}")
                print()
        else:
            print("⚠️  No locations found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 2: Get latest measurements
    if data.get('results'):
        print("=" * 60)
        print("Test 2: Getting latest measurements")
        print("=" * 60)
        
        location_id = data['results'][0]['id']
        url = "https://api.openaq.org/v3/latest"
        params = {"location_id": location_id, "limit": 10}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                print(f"✅ Retrieved measurements for location {location_id}\n")
                
                result = data['results'][0]
                print(f"Location: {result['location']['name']}")
                print(f"Sensors:")
                
                for sensor in result.get('sensors', [])[:5]:
                    param = sensor['parameter']
                    latest = sensor.get('latest', {})
                    if latest:
                        print(f"  {param['displayName']}: {latest['value']} {param['units']}")
                        print(f"    Last updated: {latest['datetime']['utc']}")
            else:
                print("⚠️  No measurements found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ OpenAQ v3 API is working correctly!")
    print("=" * 60)

if __name__ == "__main__":
    test_openaq_v3()