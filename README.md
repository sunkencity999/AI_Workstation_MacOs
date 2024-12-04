# Shell-GPT Interface

An advanced AI-powered document and system interaction tool with comprehensive file analysis and web search capabilities.

## Features

- **Multi-Model Support**
  - OpenAI GPT-4 and GPT-3.5
  - Local Ollama models (including vision models)
  - Automatic model switching and fallback

- **Document Analysis**
  - PDF documents (both simple and complex)
  - Word documents (.docx)
  - Text files (.txt, .rtf)
  - Images (PNG, JPG, JPEG)
  - Vision analysis with GPT-4 Vision or Ollama vision models

- **Advanced Workspace Analysis**
  - Comprehensive code metrics and statistics
    - Lines of code, comment density, function counts
    - File size distribution and complexity metrics
    - Documentation coverage analysis
    - Test coverage tracking
  - Multi-language support
    - Python, JavaScript/TypeScript
    - Java, C++
    - Language-specific dependency analysis
  - Code quality insights
    - Automated improvement suggestions
    - Structure and complexity analysis
    - Documentation completeness checks
  - Project-wide analytics
    - Dependency tracking across languages
    - Test coverage distribution
    - Documentation status overview

- **Interactive Interface**
  - Clean, modern Streamlit UI
  - Dark theme support
  - File upload and preview
  - Copy to clipboard functionality
  - Clear response button
  - Expandable sections in sidebar

- **Web Search Integration**
  - DuckDuckGo search support
  - Web page content analysis
  - Search result summarization

- **Email Integration**
  - Robust email functionality using nmail configuration with direct SMTP support

## Email Integration

The system includes robust email functionality using nmail configuration with direct SMTP support.

### Email Setup

1. Install nmail:
```bash
brew install nmail
```

2. Configure nmail for your email provider:
```bash
# For Gmail
nmail -s gmail
```
This will create configuration files in `~/.config/nmail/`.

3. Gmail-specific setup:
   - Enable 2-Factor Authentication in your Google Account
   - Generate an App Password:
     1. Go to Google Account Settings
     2. Security → 2-Step Verification
     3. App Passwords (at the bottom)
     4. Select "Mail" and your device
     5. Use the generated 16-character password
   - The App Password should be stored in the nmail configuration

### Configuration Files

The system uses three main configuration files in `~/.config/nmail/`:

1. `main.conf` - Primary configuration file containing:
   ```
   address=your.email@gmail.com
   name=Your Name
   smtp_host=smtp.gmail.com
   smtp_port=587
   user=your.email@gmail.com
   ```

2. `auth.conf` - Authentication configuration
3. `ui.conf` - UI preferences (optional)

### Using Email Functionality

The email system supports:
- Sending emails with subject and body
- Proper MIME message formatting
- Secure SMTP connections with TLS
- Error handling and logging

Example usage in code:
```python
from email_manager import EmailManager

manager = EmailManager()
success, message = manager.send_email(
    recipient="recipient@example.com",
    subject="Test Subject",
    body="Email body content"
)
```

### Troubleshooting

Common issues and solutions:

1. Authentication Errors:
   - Verify your App Password is correct
   - Ensure 2FA is enabled for Gmail
   - Check the password is properly stored in nmail config

2. Configuration Issues:
   - Verify all required fields in `main.conf`
   - Check SMTP settings match your provider
   - Ensure proper file permissions on config files

3. Connection Problems:
   - Verify internet connectivity
   - Check if SMTP port is not blocked
   - Ensure TLS/SSL is properly configured

### Security Notes

- Never store email passwords in plain text
- Use environment variables or secure storage for sensitive data
- Regularly update App Passwords
- Monitor email sending logs for unauthorized usage

## System Requirements

- Python 3.8 or higher
- macOS (for system commands)
- Homebrew (for system dependencies)

## Installation

1. Install system dependencies:
```bash
brew install libmagic
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
- OpenAI API key (optional if using Ollama)
- Can be set via environment variable: `OPENAI_API_KEY`
- Or stored in `~/.config/shell-gpt/.sgptrc`

## Usage

1. Start the application:
```bash
streamlit run sgpt_interface.py
```

2. Select your preferred model:
- Toggle between OpenAI and Ollama models
- Choose specific model from dropdown

3. Upload documents:
- Drag and drop or click to upload
- Supports PDF, Word, text files, and images
- Multiple file analysis
- Improved handling of both simple and complex PDF files
- Support for both file paths and bytes input for DOCX files

4. Enter your prompt:
- Direct questions
- Document analysis requests
- Image analysis queries
- Web search queries (prefix with "search:")
- Code analysis commands:
  - "analyze workspace": Get comprehensive codebase metrics
  - "analyze file [path]": Get detailed file analysis
  - "suggest improvements [path]": Get code improvement suggestions

5. Workspace Analysis Features:
- **Code Metrics**
  - View file size distribution
  - Check documentation coverage
  - Monitor test coverage
  - Track code complexity
- **Quality Insights**
  - Receive targeted improvement suggestions
  - Identify areas needing documentation
  - Find complex code sections
  - Track TODO comments
- **Dependencies**
  - View project dependencies
  - Check import usage
  - Identify heavily dependent modules

6. Generate and manage responses:
- Click "Generate Response"
- Copy results to clipboard
- Clear responses as needed

## Best Practices

- Use specific prompts for better results
- Prefix web searches with "search:"
- Choose appropriate models for different tasks:
  - GPT-4 or Ollama vision models for images
  - Standard models for text analysis
- Keep documents under reasonable size limits

## Development and Testing

The project includes a comprehensive test suite using pytest. To run the tests:

```bash
python -m pytest tests/ -v
```

Key test areas include:
- API key management
- Shell command execution
- Python script execution
- Ollama server status checks
- Web search functionality
- File processing (PDF and DOCX)
- Image processing

## Developer Information

- Author: Christopher Bradford
- Contact: contact@christopherdanielbradford.com
- License: MIT

## Troubleshooting

- Ensure all dependencies are installed
- Check API key configuration
- Verify Ollama is running for local models
- Confirm file permissions for uploads

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.