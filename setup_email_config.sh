#!/bin/bash

# Copy config template
cp config.yaml.template config.yaml

# Add environment variable to shell config if it doesn't exist
if ! grep -q "export NMAIL_PASSWORD=" ~/.zshrc; then
    echo 'export NMAIL_PASSWORD="eztf thdt uuoz sqxn"' >> ~/.zshrc
    echo "Added NMAIL_PASSWORD to ~/.zshrc"
fi

# Set environment variable for current session
export NMAIL_PASSWORD="eztf thdt uuoz sqxn"

echo "Email configuration setup complete!"
echo "1. config.yaml has been created"
echo "2. NMAIL_PASSWORD has been added to ~/.zshrc"
echo "3. NMAIL_PASSWORD has been set for current session"
echo ""
echo "Please restart your terminal or run:"
echo "source ~/.zshrc"
