// API configuration
const API_URL = 'http://localhost:8000/generate-queries';

/**
 * Switches the active dashboard view based on the provided dashboard ID
 * @param {string} dashboardId - The ID prefix of the dashboard to show ('user', 'hr', 'admin', 'licensing')
 */
function switchDashboard(dashboardId) {
    // 1. Update navigation buttons
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.classList.remove('active');
        // Simple heuristic to match button to ID, in a real app use data attributes
        if (item.textContent.toLowerCase().includes(dashboardId) ||
            (dashboardId === 'user' && item.textContent.includes('User')) ||
            (dashboardId === 'hr' && item.textContent.includes('HR')) ||
            (dashboardId === 'admin' && item.textContent.includes('Admin')) ||
            (dashboardId === 'licensing' && item.textContent.includes('Licensing'))) {
            item.classList.add('active');
        }
    });

    // 2. Update dashboard views
    const views = document.querySelectorAll('.dashboard-view');
    views.forEach(view => {
        view.classList.remove('active-dashboard');
        view.classList.add('hidden');
    });

    const targetView = document.getElementById(`${dashboardId}-dashboard`);
    if (targetView) {
        targetView.classList.remove('hidden');
        targetView.classList.add('active-dashboard');
    }
}

/**
 * Fetches queries from the backend API
 */
async function generateQueries() {
    const role = document.getElementById('role').value;
    const skills = document.getElementById('skills').value;
    const experience = document.getElementById('experience').value;
    const location = document.getElementById('location').value;

    if (!role || !skills || !experience || !location) {
        showToast('Please fill in all fields');
        return;
    }

    const payload = {
        role: role,
        skills: skills,
        experience: experience,
        location: location
    };

    // UI State updates
    const btn = document.getElementById('generate-btn');
    const spinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');

    btn.disabled = true;
    spinner.classList.remove('hidden');
    resultsContainer.classList.add('hidden');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        renderQueries(data);
        showToast('Queries generated successfully!');
    } catch (error) {
        console.error('Error generating queries:', error);
        showToast('Failed to generate queries. Ensure backend is running.');
    } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
    }
}

/**
 * Renders the query results into the UI
 * @param {Object} data - The response data from the backend
 */
function renderQueries(data) {
    const container = document.getElementById('results-container');
    container.innerHTML = ''; // Clear previous results

    // The expected keys from our API
    const platforms = [
        { key: 'linkedin', name: 'LinkedIn Query' },
        { key: 'indeed', name: 'Indeed Query' },
        { key: 'naukri', name: 'Naukri Query' },
        { key: 'glassdoor', name: 'Glassdoor Query' },
        { key: 'reed', name: 'Reed Query' },
        { key: 'totaljobs', name: 'TotalJobs Query' }
    ];

    platforms.forEach(platform => {
        if (data[platform.key]) {
            const card = document.createElement('div');
            card.className = 'query-card';

            const title = document.createElement('h4');
            title.textContent = platform.name;

            const codeBlock = document.createElement('div');
            codeBlock.className = 'query-code';
            codeBlock.id = `code-${platform.key}`;
            codeBlock.textContent = data[platform.key];

            const copyBtn = document.createElement('button');
            copyBtn.className = 'btn-outline copy-btn';
            copyBtn.textContent = 'Copy';
            copyBtn.onclick = () => copyToClipboard(data[platform.key]);

            card.appendChild(title);
            card.appendChild(codeBlock);
            card.appendChild(copyBtn);

            container.appendChild(card);
        }
    });

    container.classList.remove('hidden');
}

/**
 * Copies text to the clipboard
 * @param {string} text - The text to copy
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        showToast('Failed to copy text');
    });
}

/**
 * Shows a toast notification message
 * @param {string} message - The message to display
 */
function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.remove('hidden');

    // Hide after 3 seconds
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}
