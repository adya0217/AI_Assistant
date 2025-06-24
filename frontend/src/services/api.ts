import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add request interceptor for logging
api.interceptors.request.use(request => {
    console.log('🚀 API Request:', {
        url: request.url,
        method: request.method,
        headers: request.headers,
        data: request.data
    });
    return request;
});

// Add response interceptor for logging
api.interceptors.response.use(
    response => {
        console.log('✅ API Response:', {
            url: response.config.url,
            status: response.status,
            data: response.data
        });
        return response;
    },
    error => {
        console.error('❌ API Error:', {
            url: error.config?.url,
            status: error.response?.status,
            message: error.message,
            data: error.response?.data
        });
        throw error;
    }
);

export const sendMessage = async (message: string) => {
    try {
        console.log('📝 Sending text message:', message);
        const response = await api.post('/ask_text', { query: message });
        console.log('📥 Received text response:', response.data);
        return {
            message: response.data.response,
            timestamp: new Date().toLocaleTimeString()
        };
    } catch (error) {
        console.error('❌ Error sending message:', error);
        throw error;
    }
};

export const uploadFile = async (file: File, type: 'audio' | 'image') => {
    try {
        console.log(`📤 Starting ${type} file upload:`, {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type
        });

        const formData = new FormData();
        formData.append('file', file);

        const endpoint = type === 'audio' ? '/ask/voice' : '/ask/image';
        console.log(`📍 Uploading to endpoint: ${endpoint}`);

        const response = await api.post(endpoint, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        console.log(`📥 Received ${type} upload response:`, response.data);
        return {
            message: response.data.response,
            timestamp: new Date().toLocaleTimeString()
        };
    } catch (error) {
        console.error(`❌ Error uploading ${type} file:`, error);
        throw error;
    }
};

// Analyze classroom image using the new unified endpoint
export const analyzeImage = async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:8000/api/unified/analyze-image', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to analyze image' }));
        throw new Error(errorData.detail);
    }

    return response.json();
};

export const analyzeImageWithText = async (file: File, text?: string): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    if (text) formData.append('text', text);
    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to analyze image' }));
        throw new Error(errorData.detail);
    }
    return response.json();
};

export default api; 