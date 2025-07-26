# Currency Converter UI

A modern React application that allows users to upload currency images and convert them to Nigerian Naira (₦). The application sends image files to a backend API for processing and displays the conversion results.

## Features

- 🖼️ **Image Upload**: Drag and drop or click to upload currency images
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations
- 🔄 **Real-time Processing**: Live feedback during image processing
- 📊 **Detailed Results**: Shows original currency, amount, conversion rate, and confidence
- ⚙️ **Configurable Backend**: Easy to configure backend URL
- 📷 **Camera Support**: Ready for camera integration (placeholder)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## Prerequisites

- Node.js (version 14 or higher)
- npm or yarn package manager
- A running backend API server

## Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd currency-converter-ui
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm start
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Backend API Requirements

The application expects a backend API with the following endpoint:

### POST `/convert-currency`

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with an `image` field containing the uploaded file

**Example Request:**
```javascript
const formData = new FormData();
formData.append('image', file);

fetch('http://localhost:8000/convert-currency', {
  method: 'POST',
  body: formData
});
```

**Expected Response:**
```json
{
  "originalCurrency": "USD",
  "originalAmount": "100.00",
  "convertedAmount": "45000.00",
  "exchangeRate": "450.00",
  "confidence": 0.95
}
```

**Response Fields:**
- `originalCurrency` (string): The detected currency code (e.g., "USD", "EUR")
- `originalAmount` (string): The detected amount in original currency
- `convertedAmount` (string): The converted amount in Nigerian Naira
- `exchangeRate` (string): The exchange rate used for conversion
- `confidence` (number, optional): Confidence score of the detection (0-1)

## Configuration

### Backend URL

You can configure the backend URL in the application:

1. Open the application in your browser
2. Enter your backend URL in the "Backend URL" field
3. The default is `http://localhost:8000`

## Usage

1. **Upload Image**: 
   - Drag and drop an image onto the upload area, or
   - Click the upload area to browse and select a file

2. **Configure Backend**: 
   - Enter your backend API URL if different from default

3. **Convert Currency**: 
   - Click "Convert to Naira" button
   - Wait for processing (usually takes a few seconds)

4. **View Results**: 
   - See the conversion details including original currency, amount, and Naira equivalent
   - Use "Convert Another Image" to process more images

## Error Handling

The application handles various error scenarios:

- **Invalid file type**: Shows error for non-image files
- **Network errors**: Displays connection issues
- **Backend errors**: Shows server-side error messages
- **Timeout**: 30-second timeout for API requests

## Development

### Project Structure

```
src/
├── App.js          # Main application component
├── App.css         # Component-specific styles
├── index.js        # Application entry point
└── index.css       # Global styles
```

### Available Scripts

- `npm start`: Runs the app in development mode
- `npm build`: Builds the app for production
- `npm test`: Launches the test runner
- `npm eject`: Ejects from Create React App (irreversible)

### Customization

You can customize the application by:

1. **Styling**: Modify `src/index.css` for global styles or `src/App.css` for component styles
2. **API Integration**: Update the axios configuration in `src/App.js`
3. **Features**: Add new functionality by extending the React components

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure your backend allows requests from the frontend domain
2. **Image Upload Fails**: Check file size and format restrictions
3. **Backend Connection**: Verify the backend URL and server status

### Debug Mode

Open browser developer tools (F12) to see detailed error messages and network requests.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review browser console for error messages
3. Verify backend API is running and accessible 