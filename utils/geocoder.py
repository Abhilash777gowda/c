import re
from typing import Dict, Optional, Tuple

# Comprehensive dictionary of major Indian cities and states with coordinates
# Compiled for CRIMSON-India real-time mapping
INDIAN_CITIES = {
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    # Kannada city names
    "ಬೆಂಗಳೂರು": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "ಮೈಸೂರು": (12.2958, 76.6394),
    "ಹುಬ್ಬಳ್ಳಿ": (15.3647, 75.1240),
    "ಧಾರವಾಡ": (15.3647, 75.1240),
    "ಮಂಗಳೂರು": (12.9141, 74.8560),
    "mangalore": (12.9141, 74.8560),
    "belgaum": (15.8497, 74.4977),
    "belagavi": (15.8497, 74.4977),
    "ಬೆಳಗಾವಿ": (15.8497, 74.4977),
    "gulbarga": (17.3297, 76.8343),
    "kalaburagi": (17.3297, 76.8343),
    "ಕಲಬುರಗಿ": (17.3297, 76.8343),
    "davanagere": (14.4644, 75.9218),
    "ದಾವಣಗೆರೆ": (14.4644, 75.9218),
    "bellary": (15.1394, 76.9214),
    "ballari": (15.1394, 76.9214),
    "ಬಳ್ಳಾರಿ": (15.1394, 76.9214),
    "shimoga": (13.9299, 75.5681),
    "shivamogga": (13.9299, 75.5681),
    "ಶಿವಮೊಗ್ಗ": (13.9299, 75.5681),
    "tumkur": (13.3392, 77.1140),
    "tumakuru": (13.3392, 77.1140),
    "ತುಮಕೂರು": (13.3392, 77.1140),
    "raichur": (16.2120, 77.3559),
    "ರಾಯಚೂರು": (16.2120, 77.3559),
    "bidar": (17.9104, 77.5199),
    "ಬೀದರ್": (17.9104, 77.5199),
    "hospet": (15.2689, 76.3909),
    "hosapete": (15.2689, 76.3909),
    "ಹೊಸಪೇಟೆ": (15.2689, 76.3909),
    "ahmedabad": (23.0225, 72.5714),
    "pune": (18.5204, 73.8567),
    "surat": (21.1702, 72.8311),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "nagpur": (21.1458, 79.0882),
    "indore": (22.7196, 75.8577),
    "thane": (19.2183, 72.9781),
    "bhopal": (23.2599, 77.4126),
    "visakhapatnam": (17.6868, 83.2185),
    "pimpri-chinchwad": (18.6298, 73.7997),
    "patna": (25.5941, 85.1376),
    "vadodara": (22.3072, 73.1812),
    "ghaziabad": (28.6692, 77.4538),
    "ludhiana": (30.9010, 75.8573),
    "agra": (27.1767, 78.0081),
    "nashik": (19.9975, 73.7898),
    "faridabad": (28.4089, 77.3178),
    "meerut": (28.9845, 77.7064),
    "rajkot": (22.3039, 70.8022),
    "kalyan-dombivli": (19.2437, 73.1352),
    "vasai-virar": (19.3919, 72.8397),
    "varanasi": (25.3176, 82.9739),
    "srinagar": (34.0837, 74.7973),
    "aurangabad": (19.8762, 75.3433),
    "dhanbad": (23.7957, 86.4304),
    "amritsar": (31.6340, 74.8723),
    "navi mumbai": (19.0330, 73.0297),
    "allahabad": (25.4358, 81.8463),
    "prayagraj": (25.4358, 81.8463),
    "ranchi": (23.3441, 85.3096),
    "howrah": (22.5958, 88.2636),
    "coimbatore": (11.0168, 76.9558),
    "jabalpur": (23.1815, 79.9864),
    "gwalior": (26.2124, 78.1772),
    "vijayawada": (16.5062, 80.6480),
    "jodhpur": (26.2389, 73.0243),
    "madurai": (9.9252, 78.1198),
    "raipur": (21.2514, 81.6296),
    "kota": (25.2138, 75.8648),
    "guwahati": (26.1445, 91.7362),
    "chandigarh": (30.7333, 76.7794),
    "solapur": (17.6599, 75.9064),
    "hubli-dharwad": (15.3647, 75.1240),
    "bareilly": (28.3670, 79.4304),
    "moradabad": (28.8351, 78.7733),
    "mysore": (12.2958, 76.6394),
    "mysuru": (12.2958, 76.6394),
    "gurgaon": (28.4595, 77.0266),
    "gurugram": (28.4595, 77.0266),
    "aligarh": (27.8974, 78.0880),
    "jalandhar": (31.3260, 75.5762),
    "tiruchirappalli": (10.7905, 78.7047),
    "bhubaneswar": (20.2961, 85.8245),
    "salem": (11.6643, 78.1460),
    "mira-bhayandar": (19.2813, 72.8557),
    "warangal": (17.9689, 79.5941),
    "thiruvananthapuram": (8.5241, 76.9366),
    "trivandrum": (8.5241, 76.9366),
    "bhiwandi": (19.2813, 73.0483),
    "saharanpur": (29.9640, 77.5460),
    "guntur": (16.3067, 80.4365),
    "amravati": (20.9320, 77.7523),
    "noida": (28.5355, 77.3910),
    "jamshedpur": (22.8046, 86.2029),
    "bhilai": (21.1938, 81.3509),
    "cuttack": (20.4625, 85.8830),
    "kochi": (9.9312, 76.2673),
    "udaipur": (24.5854, 73.7125),
    "dehradun": (30.3165, 78.0322),
    "shimla": (31.1048, 77.1734),
    "jammu": (32.7266, 74.8570),
    "mangalore": (12.9141, 74.8560),
    "mangalur": (12.9141, 74.8560),
    "belgaum": (15.8497, 74.4977),
    "belagavi": (15.8497, 74.4977),
    "uprooted": (20.5937, 78.9629), # Generic fallback placeholder
    "mandya": (12.5218, 76.8951),
    "ಮಂಡ್ಯ": (12.5218, 76.8951),
    "hassan": (13.0068, 76.1020),
    "ಹಾಸನ": (13.0068, 76.1020),
    "chitradurga": (14.2251, 76.3980),
    "ಚಿತ್ರದುರ್ಗ": (14.2251, 76.3980),
    "kolar": (13.1367, 78.1292),
    "ಕೋಲಾರ": (13.1367, 78.1292),
    "tumakuru": (13.3392, 77.1140),
    "chikmagalur": (13.3153, 75.7754),
    "chikkamagaluru": (13.3153, 75.7754),
    "ಉಡುಪಿ": (13.3409, 74.7421),
    "udupi": (13.3409, 74.7421),
    "kodagu": (12.4244, 75.7382),
    "coorg": (12.4244, 75.7382),
}

def extract_location(text: str) -> Optional[str]:
    """
    Extract a city name from the text using the predefined dictionary.
    Returns the city name if found, else None.
    """
    if not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    # Sort cities by length descending to match "Navi Mumbai" before "Mumbai"
    sorted_cities = sorted(INDIAN_CITIES.keys(), key=len, reverse=True)
    
    for city in sorted_cities:
        # Match whole word to avoid "Indore" matching "In"
        pattern = rf'\b{re.escape(city)}\b'
        if re.search(pattern, text_lower):
            return city
            
    return None

def geocode_location(location_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (lat, lon) for a given city name.
    """
    if not location_name:
        return None, None
        
    return INDIAN_CITIES.get(location_name.lower(), (None, None))
