# Hospital Chatbot

A conversational AI-powered chatbot application designed for hospital management, allowing patients to book appointments, inquire about doctors and vaccines, and interact with an intelligent assistant for medical-related queries.

## Features

- **Appointment Booking**: Schedule appointments with available doctors based on their specialization and availability.
- **Doctor Information**: Get details about doctors, their specializations, and available hours.
- **Vaccine Management**: Check vaccine stock and availability.
- **AI-Powered Chat**: Uses Hugging Face models for natural language processing and conversation.
- **Web Interface**: User-friendly web UI for interacting with the chatbot.
- **Database Integration**: Stores data in PostgreSQL (or SQLite for local development).

## Tech Stack

- **Backend**: Python 3.11+, FastAPI
- **Database**: SQLAlchemy ORM with PostgreSQL (production) or SQLite (development)
- **AI/ML**: Hugging Face API for chat completions
- **Frontend**: HTML, CSS (static files served by FastAPI)
- **Deployment**: Render (with render.yaml configuration)
- **Dependencies**: See `requirements.txt`

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database (for production; SQLite works for local testing)
- Hugging Face account and API token (for AI features)
- Virtual environment (recommended)

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment**:
   ```bash
   python -m venv chatbotvenv
   ```

3. **Activate the virtual environment**:
   - On Windows: `chatbotvenv\Scripts\activate`
   - On macOS/Linux: `source chatbotvenv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables** (optional for local development):
   - `DATABASE_URL`: Database connection string (defaults to SQLite)
   - `HF_TOKEN`: Your Hugging Face API token
   - `HF_MODELS`: Comma-separated list of Hugging Face models (defaults provided)
   - `PORT`: Port for the server (defaults to 8000)

6. **Initialize the database**:
   The database is automatically initialized and seeded with sample data when the app starts. You can also run:
   ```bash
   python populate.py
   ```

## Usage

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:8000/static/index.html`

3. **Interact with the chatbot**:
   - Type messages in the chat interface to book appointments, ask about doctors, or inquire about vaccines.
   - The chatbot uses AI to understand and respond to natural language queries.

### API Endpoints

The application exposes the following REST API endpoints:

- `GET /doctors`: Retrieve list of doctors
- `GET /vaccines`: Retrieve list of vaccines
- `GET /appointments`: Retrieve list of appointments
- `POST /chat`: Send a chat message and receive a response
- `POST /appointments`: Create a new appointment
- `GET /static/index.html`: Serve the web interface

Example API usage with curl:
```bash
curl -X GET "http://localhost:8000/doctors"
```

## Database Schema

- **Doctors**: id, name, specialization, available_time
- **Vaccines**: id, name, stock
- **Appointments**: id, patient_name, doctor_id, vaccine_id (optional), date_time

Sample data is automatically seeded on startup.

## Testing

### Manual Testing

1. Start the server as described in Usage.
2. Use the web interface to test chat functionality:
   - Book an appointment: "I want to book an appointment with Dr. Ramesh Kumar for tomorrow at 11 AM"
   - Ask about doctors: "What doctors are available?"
   - Check vaccines: "Do you have MMR vaccine in stock?"

3. Test API endpoints using tools like Postman or curl.

### Automated Testing

Currently, no automated tests are implemented. To add tests:

1. Create a `tests/` directory.
2. Use `pytest` for unit and integration tests.
3. Test database operations, API endpoints, and chatbot logic.

Example test structure:
```
tests/
├── test_api.py
├── test_chatbot.py
└── test_db.py
```

## Deployment

The application is configured for deployment on Render using `render.yaml`:

1. **Database**: A PostgreSQL database is automatically provisioned.
2. **Web Service**: The FastAPI app is deployed as a web service.
3. **Environment Variables**: Set `HF_TOKEN` in Render's environment settings.

To deploy:
1. Connect your GitHub repository to Render.
2. Use the `render.yaml` configuration.
3. Set environment variables in Render dashboard.

## Standard Operating Procedures (SOPs)

### Development Workflow

1. Work on features in a separate branch.
2. Test locally before committing.
3. Use meaningful commit messages.
4. Create pull requests for code review.

### Database Management

- **Local**: Uses SQLite by default.
- **Production**: Uses PostgreSQL on Render.
- **Migrations**: No migration system implemented; schema changes require manual handling.

### Monitoring and Maintenance

- Monitor server logs on Render.
- Check database connections and performance.
- Update dependencies regularly.
- Backup database data as needed.

### Security Considerations

- Store API keys securely (use environment variables).
- Validate user inputs to prevent injection attacks.
- Implement authentication if expanding to multi-user system.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test thoroughly.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues or questions, please open an issue in the repository or contact the development team.