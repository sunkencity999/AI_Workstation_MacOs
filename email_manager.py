import subprocess
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import configparser
import smtplib
import time
import yaml
from string import Template

class EmailManager:
    def __init__(self):
        """Initialize email manager."""
        self.config_dir = Path.home() / '.config' / 'nmail'
        self.main_config_file = self.config_dir / 'main.conf'
        self.auth_config_file = self.config_dir / 'auth.conf'
        self.ui_config_file = self.config_dir / 'ui.conf'
        self.yaml_config_file = Path('config.yaml')
        
        # Set up logging
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'email_manager.log'
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def is_nmail_installed(self) -> bool:
        """Check if nmail is installed."""
        try:
            result = subprocess.run(['which', 'nmail'], 
                                 capture_output=True, 
                                 text=True)
            return bool(result.stdout.strip())
        except Exception as e:
            logging.error(f"Error checking nmail installation: {str(e)}")
            return False
            
    def is_nmail_configured(self) -> bool:
        """Check if nmail is configured."""
        if not self.main_config_file.exists():
            logging.error("nmail main.conf does not exist")
            return False
            
        try:
            # Read the config file and parse key=value pairs
            config = {}
            with open(self.main_config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            
            # Check for essential configurations
            required_fields = [
                'address',
                'user',
                'imap_host',
                'smtp_host',
                'smtp_port',
                'name'
            ]
            
            for field in required_fields:
                if field not in config:
                    logging.error(f"nmail config missing required field: {field}")
                    return False
                    
            # Try to connect to SMTP server to verify configuration
            smtp_host = config['smtp_host']
            smtp_port = int(config['smtp_port'])
            
            try:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=5) as smtp:
                    smtp.ehlo()
                    if smtp_port == 587:
                        smtp.starttls()
                        smtp.ehlo()
                logging.info("SMTP connection test successful")
                return True
            except Exception as e:
                logging.error(f"SMTP connection test failed: {str(e)}")
                return False
                
        except Exception as e:
            logging.error(f"Error checking nmail configuration: {str(e)}")
            return False
            
    def get_supported_services(self) -> List[str]:
        """Get list of supported email services."""
        return ['gmail', 'gmail-oauth2', 'icloud', 'outlook', 'outlook-oauth2']
        
    def setup_email_service(self, service: str) -> Tuple[bool, str]:
        """Run the nmail setup wizard for a specific service."""
        if service not in self.get_supported_services():
            return False, f"Unsupported service: {service}"
            
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            if service == 'gmail':
                # Create main.conf with Gmail settings
                main_config = """# Gmail Configuration
address = {email}
user = {email}
name = {name}
imap_host = imap.gmail.com
imap_port = 993
smtp_host = smtp.gmail.com
smtp_port = 587
drafts = [Gmail]/Drafts
sent = [Gmail]/Sent Mail
trash = [Gmail]/Trash
inbox = INBOX
"""
                # Write instructions file
                instructions_path = self.config_dir / 'gmail_setup.txt'
                instructions = """Gmail Account Setup Instructions:

1. First, you need to enable "Less secure app access" OR create an App Password:
   
   Option A - App Password (Recommended if you have 2-factor authentication):
   a. Go to https://myaccount.google.com/security
   b. Under "Signing in to Google", select "2-Step Verification"
   c. At the bottom, select "App passwords"
   d. Generate a new App password for "Mail" and "Other (Custom name)"
   e. Use this 16-character password when prompted by nmail
   
   Option B - Less secure app access:
   a. Go to https://myaccount.google.com/security
   b. Turn on "Less secure app access"
   
2. When nmail launches:
   a. Enter your full Gmail address when prompted
   b. Enter your Google account password or App Password
   c. Enter your full name as you want it to appear in emails
   
Press Enter in the terminal when you're ready to start the setup process."""
                
                with open(instructions_path, 'w') as f:
                    f.write(instructions)
                
                # Show instructions and launch setup
                subprocess.run(['cat', str(instructions_path)])
                input("\nPress Enter when you're ready to continue...")
                
                # Start the interactive setup
                email = input("\nEnter your Gmail address: ").strip()
                name = input("Enter your full name: ").strip()
                
                # Write the main configuration
                with open(self.main_config_file, 'w') as f:
                    f.write(main_config.format(email=email, name=name))
                
                # Launch nmail to complete the setup
                process = subprocess.Popen(['nmail'],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
                
                return True, "Gmail configuration completed. Please check your email in nmail."
                
            elif service == 'gmail-oauth2':
                # For OAuth2, we need to use the -s gmail-oauth2 flag
                process = subprocess.Popen(['nmail', '-s', 'gmail-oauth2'],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
                return True, "Gmail OAuth2 setup wizard launched. Please follow the prompts in the terminal."
                
            else:
                # For other services, use the standard setup wizard
                process = subprocess.Popen(['nmail', '-s', service],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
                return True, f"Setup wizard launched for {service}. Please follow the prompts in the terminal."
                
        except Exception as e:
            return False, f"Error running setup wizard: {str(e)}"
            
    def install_nmail(self) -> Tuple[bool, str]:
        """Install nmail using Homebrew."""
        try:
            # First check if Homebrew is installed
            brew_check = subprocess.run(['which', 'brew'], 
                                     capture_output=True, 
                                     text=True)
            if not brew_check.stdout.strip():
                return False, "Homebrew is not installed. Please install Homebrew first."
            
            # Install nmail
            result = subprocess.run(['brew', 'install', 'nmail'],
                                 capture_output=True,
                                 text=True)
            
            if result.returncode == 0:
                return True, "nmail installed successfully"
            else:
                return False, f"Error installing nmail: {result.stderr}"
                
        except Exception as e:
            return False, f"Error during installation: {str(e)}"
            
    def launch_nmail(self) -> Tuple[bool, str]:
        """Launch nmail in a new terminal window."""
        try:
            if not self.is_nmail_configured():
                return False, "nmail is not configured. Please set up an email service first."
                
            # Use AppleScript to open a new terminal window and run nmail
            apple_script = '''
            tell application "Terminal"
                do script "nmail"
                activate
            end tell
            '''
            subprocess.run(['osascript', '-e', apple_script])
            return True, "nmail launched successfully"
        except Exception as e:
            return False, f"Error launching nmail: {str(e)}"
            
    def _load_yaml_config(self) -> Optional[dict]:
        """Load configuration from YAML file with environment variable substitution."""
        if not self.yaml_config_file.exists():
            logging.warning(f"Config file {self.yaml_config_file} not found")
            return None
            
        try:
            with open(self.yaml_config_file) as f:
                # Read the file content
                content = f.read()
                
                # Replace environment variables
                template = Template(content)
                expanded = template.safe_substitute(os.environ)
                
                # Parse YAML
                config = yaml.safe_load(expanded)
                return config
        except Exception as e:
            logging.error(f"Error loading YAML config: {e}")
            return None
            
    def get_email_password(self) -> Optional[str]:
        """Get email password from environment variable or config."""
        # First try nmail config since we know it has the correct password
        try:
            with open(self.main_config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('pass='):
                        # Remove quotes if present
                        password = line.split('=', 1)[1].strip().strip('"')
                        return password
        except Exception as e:
            logging.error(f"Error reading password from nmail config: {e}")
            
        # Then try environment variable
        password = os.getenv('NMAIL_PASSWORD')
        if password:
            return password
            
        # Finally try YAML config
        config = self._load_yaml_config()
        if config and 'email' in config:
            email_config = config['email']
            if 'user' in email_config and 'password' in email_config['user']:
                return email_config['user']['password']
            
        return None

    def get_smtp_settings(self) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
        """Get SMTP settings from config."""
        # First try nmail config since we know it has the correct Gmail settings
        smtp_host = None
        smtp_port = None
        sender_email = None
        sender_name = None
        
        try:
            config = {}
            with open(self.main_config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip().strip('"')
                        config[key.strip()] = value
                        
            smtp_host = config.get('smtp_host')
            smtp_port = int(config.get('smtp_port', 0)) if config.get('smtp_port') else None
            sender_email = config.get('address')
            sender_name = config.get('name')
            
            if all([smtp_host, smtp_port, sender_email]):
                return smtp_host, smtp_port, sender_email, sender_name
                        
        except Exception as e:
            logging.error(f"Error reading SMTP settings from nmail config: {e}")
            
        # Fall back to YAML config if nmail config fails
        config = self._load_yaml_config()
        if config and 'email' in config:
            email_config = config['email']
            if all(key in email_config['smtp'] for key in ['host', 'port']):
                return (
                    email_config['smtp']['host'],
                    email_config['smtp']['port'],
                    email_config['user']['address'],
                    email_config['user'].get('name')
                )
                
        return None, None, None, None

    def send_email(self, recipient, subject, body):
        """Send an email using SMTP directly."""
        try:
            if not self.is_nmail_configured():
                return False, "nmail is not configured. Please set up an email service first."
                
            # Get SMTP settings from config
            smtp_host, smtp_port, sender_email, sender_name = self.get_smtp_settings()
            if not all([smtp_host, smtp_port, sender_email]):
                return False, "Could not read SMTP settings from nmail configuration"
                
            # Get password
            password = self.get_email_password()
            if not password:
                return False, "Email password not found. Please set up the password first."
                
            logging.info(f"Connecting to SMTP server {smtp_host}:{smtp_port}")
            
            # Create the email message
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.utils import formataddr
            
            msg = MIMEMultipart()
            msg['From'] = formataddr((sender_name or "", sender_email))
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server and send
            import smtplib
            import ssl
            
            context = ssl.create_default_context()
            
            try:
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.ehlo()  # Can be omitted
                    server.starttls(context=context)  # Secure the connection
                    server.ehlo()  # Can be omitted
                    server.login(sender_email, password)
                    server.send_message(msg)
                    
                logging.info("Email sent successfully")
                return True, "Email sent successfully"
                    
            except smtplib.SMTPAuthenticationError:
                logging.error("SMTP authentication failed")
                return False, "Email authentication failed. Please check your password."
                
            except smtplib.SMTPException as e:
                logging.error(f"SMTP error: {str(e)}")
                return False, f"SMTP error: {str(e)}"
                
        except Exception as e:
            logging.error(f"Exception while sending email: {str(e)}")
            return False, f"Exception while sending email: {str(e)}"

    def process_natural_language_command(self, command: str) -> Tuple[bool, str]:
        """Process natural language email commands."""
        if not self.is_nmail_configured():
            return False, "nmail is not configured. Please set up an email service first."
            
        try:
            # Extract email components from command
            command = command.lower()
            
            if not any(word in command for word in ['send', 'email', 'mail']):
                return False, "Command not recognized as an email command"
                
            # Extract recipient
            to = None
            if 'to ' in command:
                parts = command.split('to ')
                if len(parts) > 1:
                    to_part = parts[1].split(' ')[0]
                    if '@' in to_part:
                        to = to_part
                        
            # Extract subject
            subject = None
            if 'subject ' in command:
                parts = command.split('subject ')
                if len(parts) > 1:
                    subject = parts[1].split(' with ')[0].strip()
                    
            # Extract body
            body = None
            if 'body ' in command or 'message ' in command:
                if 'body ' in command:
                    parts = command.split('body ')
                else:
                    parts = command.split('message ')
                if len(parts) > 1:
                    body = parts[1].strip()
                    
            if not all([to, subject, body]):
                return False, "Could not extract all required email components (to, subject, body)"
                
            return self.send_email(to, subject, body)
            
        except Exception as e:
            return False, f"Error processing command: {str(e)}"

    def store_email_password(self, password: str) -> bool:
        """Store email password in keychain and environment."""
        try:
            # Store in environment
            os.environ['NMAIL_PASSWORD'] = password
            
            # Store in keychain
            subprocess.run(
                ['security', 'delete-generic-password', '-s', 'nmail_email'],
                capture_output=True
            )
            
            process = subprocess.run(
                ['security', 'add-generic-password', '-s', 'nmail_email', '-a', 'nmail', '-w', password],
                capture_output=True,
                text=True
            )
            
            # Update nmail config
            with open(self.main_config_file, 'r') as f:
                config_lines = f.readlines()
                
            with open(self.main_config_file, 'w') as f:
                for line in config_lines:
                    if not line.startswith('pass ='):
                        f.write(line)
                # Quote the password if it contains spaces
                if ' ' in password:
                    f.write(f'pass = "{password}"\n')
                else:
                    f.write(f'pass = {password}\n')
                
            return process.returncode == 0
        except Exception as e:
            logging.error(f"Failed to store password: {e}")
            return False
