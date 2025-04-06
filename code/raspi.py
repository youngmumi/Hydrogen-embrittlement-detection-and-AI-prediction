from picamera2 import Picamera2
import time
import requests

picam2 = Picamera2()


camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)


picam2.start()


time.sleep(2)
resp = requests.get('http://worldtimeapi.org/api/timezone/Asia/Seoul').json()

image_path = f"/usr/share/nginx/html/images/{resp['datetime']}.jpg"
picam2.capture_file(image_path)

print(f"Image captured and saved to {image_path}")


picam2.close()
