from twilio.rest import Client
import time
from datetime import datetime, timedelta
import logging
import sys

# Configure logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class TwilioAlert:
    def __init__(self, account_sid, auth_token, from_number):
        try:
            print("\n=== Initializing Twilio Alert System ===")
            print(f"Account SID: {account_sid}")
            print(f"From Number: {from_number}")
            
            self.client = Client(account_sid, auth_token)
            self.from_number = from_number
            self.last_alert_time = {
                'fire': datetime.min,
                'smoke': datetime.min
            }
            self.alert_cooldown = 60  # seconds between alerts
            print("Twilio Alert System initialized successfully!")
        except Exception as e:
            print(f"ERROR: Failed to initialize Twilio Alert System: {str(e)}")
            raise

    def can_send_alert(self, alert_type):
        """Check if enough time has passed since the last alert of this type"""
        current_time = datetime.now()
        time_since_last_alert = (current_time - self.last_alert_time[alert_type]).total_seconds()
        can_send = time_since_last_alert >= self.alert_cooldown
        
        print(f"\n=== Alert Status Check ===")
        print(f"Alert Type: {alert_type}")
        print(f"Time since last alert: {time_since_last_alert:.1f} seconds")
        print(f"Can send alert: {can_send}")
        
        return can_send

    def send_alert(self, to_number, alert_type, location="Unknown"):
        """Send an alert message if cooldown period has passed"""
        print(f"\n=== Attempting to send {alert_type.upper()} alert ===")
        print(f"To: {to_number}")
        print(f"From: {self.from_number}")
        print(f"Location: {location}")
        
        if not self.can_send_alert(alert_type):
            print(f"Skipping {alert_type} alert - Cooldown period not elapsed")
            return False

        try:
            message_body = f"""🚨 {alert_type.upper()} ALERT!
Location: {location}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

            print("\nSending message with body:")
            print(message_body)
            
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=to_number
            )
            
            print(f"\nMessage sent successfully!")
            print(f"Message SID: {message.sid}")
            self.last_alert_time[alert_type] = datetime.now()
            return True
            
        except Exception as e:
            print(f"\nERROR: Failed to send message")
            print(f"Error details: {str(e)}")
            print(f"From: {self.from_number}")
            print(f"To: {to_number}")
            print(f"Type: {alert_type}")
            return False 