// ObjectIQ - Profile Page JavaScript

// Profile form submission
document.getElementById('profile-form')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const data = {
        email: document.getElementById('edit-email').value,
        first_name: document.getElementById('edit-firstname').value,
        last_name: document.getElementById('edit-lastname').value,
        phone: document.getElementById('edit-phone').value,
        organization: document.getElementById('edit-org').value,
        bio: document.getElementById('edit-bio').value
    };
    
    try {
        const response = await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification('Profile updated successfully!', 'success');
        } else {
            showNotification(result.message || 'Failed to update profile', 'error');
        }
    } catch (error) {
        console.error('Update error:', error);
        showNotification('Failed to update profile', 'error');
    }
});

// Password form submission
document.getElementById('password-form')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    
    if (newPassword !== confirmPassword) {
        showNotification('Passwords do not match', 'error');
        return;
    }
    
    if (newPassword.length < 6) {
        showNotification('Password must be at least 6 characters', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/user/password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification('Password changed successfully!', 'success');
            document.getElementById('password-form').reset();
        } else {
            showNotification(result.message || 'Failed to change password', 'error');
        }
    } catch (error) {
        console.error('Password change error:', error);
        showNotification('Failed to change password', 'error');
    }
});

// Profile image upload
async function uploadProfileImage(input) {
    if (input.files && input.files[0]) {
        // Check file size (2MB limit)
        if (input.files[0].size > 2 * 1024 * 1024) {
            showNotification('Image size must be less than 2MB', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', input.files[0]);
        
        try {
            const response = await fetch('/api/user/profile/image', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'ok') {
                // Update displayed image
                const reader = new FileReader();
                reader.onload = function(e) {
                    const imgEl = document.getElementById('profile-avatar-img');
                    const iconEl = document.getElementById('profile-avatar-icon');
                    imgEl.src = e.target.result;
                    imgEl.style.display = 'block';
                    iconEl.style.display = 'none';
                };
                reader.readAsDataURL(input.files[0]);
                showNotification('Profile image updated!', 'success');
            } else {
                showNotification(data.message || 'Failed to update image', 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            showNotification('Failed to upload image', 'error');
        }
    }
}

// Delete Account Modal
function showDeleteAccountModal() {
    document.getElementById('delete-modal').classList.add('open');
}

function closeDeleteModal() {
    document.getElementById('delete-modal').classList.remove('open');
    document.getElementById('delete-confirm').value = '';
}

async function confirmDeleteAccount() {
    const confirmText = document.getElementById('delete-confirm').value;
    
    if (confirmText !== 'DELETE') {
        showNotification('Please type DELETE to confirm', 'error');
        return;
    }
    
    if (!confirm('This action cannot be undone. Are you absolutely sure?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/user/account', {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification('Account deleted successfully', 'success');
            setTimeout(() => {
                window.location.href = '/logout';
            }, 2000);
        } else {
            showNotification(result.message || 'Failed to delete account', 'error');
        }
    } catch (error) {
        console.error('Delete account error:', error);
        showNotification('Failed to delete account', 'error');
    }
}

// Toast Notifications
function showNotification(message, type = 'info') {
    const container = document.createElement('div');
    container.className = `toast-notification ${type}`;
    container.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(container);
    
    // Animate in
    setTimeout(() => container.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        container.classList.remove('show');
        setTimeout(() => container.remove(), 300);
    }, 3000);
}

// Initialize profile image if exists
document.addEventListener('DOMContentLoaded', function() {
    const profileImg = document.getElementById('profile-avatar-img');
    const profileIcon = document.getElementById('profile-avatar-icon');
    
    // Check if user has profile image (would be set from template)
    if (typeof USER_PROFILE_IMAGE !== 'undefined' && USER_PROFILE_IMAGE) {
        profileImg.src = USER_PROFILE_IMAGE;
        profileImg.style.display = 'block';
        profileIcon.style.display = 'none';
    }
});
