import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { FaMoon, FaUser, FaSignOutAlt, FaBell } from 'react-icons/fa';
import Avatar from 'react-avatar';
import useAuthStore from '../../store/authStore';
import toast from 'react-hot-toast';

export default function Navbar() {
  const [showNotifications, setShowNotifications] = useState(false);
  const { user, logout, isAuthenticated } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    toast.success('Logged out successfully');
    navigate('/login');
  };

  if (!isAuthenticated) {
    return (
      <nav className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <Link to="/" className="text-2xl font-bold text-primary-600">😴 InsomniaRisk</Link>
          <div className="space-x-4">
            <Link to="/login" className="btn-secondary">Login</Link>
            <Link to="/register" className="btn-primary">Register</Link>
          </div>
        </div>
      </nav>
    );
  }

  return (
    <nav className="bg-white shadow-sm border-b sticky top-0 z-50">
      <div className="container mx-auto px-6 py-3 flex justify-between items-center">
        <Link to="/dashboard" className="text-2xl font-bold text-primary-600">😴 InsomniaRisk</Link>
        
        <div className="flex items-center gap-4">
          <div className="relative">
            <button onClick={() => setShowNotifications(!showNotifications)} className="p-2 hover:bg-gray-100 rounded-full relative">
              <FaBell className="text-xl" />
              <span className="absolute top-0 right-0 w-2 h-2 bg-danger-500 rounded-full"></span>
            </button>
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-xl border">
                <div className="p-4 border-b">
                  <h4 className="font-bold">Notifications</h4>
                </div>
                <div className="p-4 text-center text-gray-500">No new notifications</div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <Avatar name={user?.name || 'User'} size="40" round={true} />
            <div>
              <p className="font-medium">{user?.name || 'John Doe'}</p>
              <button onClick={handleLogout} className="text-sm text-gray-500 hover:text-danger-500 flex items-center gap-1">
                <FaSignOutAlt /> Logout
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
