import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FaMoon, FaChartLine, FaHistory, FaUser, FaComments, FaBell, FaCog, FaSignOutAlt } from 'react-icons/fa';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import Avatar from 'react-avatar';
import useAuthStore from '../../store/authStore';
import toast from 'react-hot-toast';

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { user, logout, isAuthenticated } = useAuthStore();
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    toast.success('Logged out successfully');
    navigate('/login');
  };

  const menuItems = [
    { path: '/dashboard', icon: <FaChartLine />, label: 'Dashboard' },
    { path: '/predict', icon: <FaMoon />, label: 'Predict' },
    { path: '/history', icon: <FaHistory />, label: 'History' },
    { path: '/chat', icon: <FaComments />, label: 'Chat' },
    { path: '/profile', icon: <FaUser />, label: 'Profile' },
    { path: '/settings', icon: <FaCog />, label: 'Settings' }
  ];

  const getLinkClass = (path) => {
    const baseClass = 'flex items-center gap-4 p-3 rounded-lg transition';
    const activeClass = location.pathname === path ? 'bg-primary-600' : 'hover:bg-gray-800';
    return baseClass + ' ' + activeClass;
  };

  if (!isAuthenticated) {
    return null; // Don't show sidebar if not authenticated
  }

  return (
    <motion.div 
      animate={{ width: collapsed ? 80 : 280 }}
      className="bg-gray-900 text-white h-screen sticky top-0 overflow-hidden flex flex-col"
    >
      <div className="p-6 flex-1">
        <div className="flex items-center justify-between mb-8">
          {!collapsed && <span className="text-2xl font-bold">😴 Insomnia</span>}
          <button onClick={() => setCollapsed(!collapsed)} className="p-2 hover:bg-gray-800 rounded-lg">
            <FaBell />
          </button>
        </div>

        <div className="space-y-2">
          {menuItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={getLinkClass(item.path)}
            >
              <span className="text-xl">{item.icon}</span>
              {!collapsed && <span>{item.label}</span>}
            </Link>
          ))}
        </div>
      </div>

      {/* User Profile & Logout - Always at bottom */}
      <div className="p-6 border-t border-gray-800">
        <div className="flex items-center gap-3">
          <Avatar name={user?.name || 'User'} size="40" round={true} />
          {!collapsed && (
            <div className="flex-1">
              <p className="font-medium truncate">{user?.name || 'John Doe'}</p>
              <button 
                onClick={handleLogout}
                className="text-sm text-gray-400 hover:text-white flex items-center gap-1 transition-colors"
              >
                <FaSignOutAlt size={12} /> Logout
              </button>
            </div>
          )}
          {collapsed && (
            <button 
              onClick={handleLogout}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              title="Logout"
            >
              <FaSignOutAlt className="text-gray-400 hover:text-white" />
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}
