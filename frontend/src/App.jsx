import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import useAuthStore from './store/authStore';

// Layout
import Navbar from './components/Layout/Navbar';
import Sidebar from './components/Layout/Sidebar';

// Pages
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import History from './pages/History';
import Profile from './pages/Profile';
import Chat from './pages/Chat';
import Settings from './pages/Settings';

// Styles
import './styles/globals.css';

const ProtectedRoute = ({ children }) => {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  return isAuthenticated ? children : <Navigate to='/login' />;
};

function App() {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  return (
    <BrowserRouter>
      <div className='min-h-screen flex'>
        {isAuthenticated && <Sidebar />}
        <div className='flex-1 flex flex-col'>
          <Navbar />
          <main className='flex-1 container mx-auto px-6 py-8 max-w-7xl'>
            <AnimatePresence mode='wait'>
              <Routes>
                <Route path='/' element={<Home />} />
                <Route path='/login' element={<Login />} />
                <Route path='/register' element={<Register />} />
                <Route path='/dashboard' element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
                <Route path='/predict' element={<ProtectedRoute><Predict /></ProtectedRoute>} />
                <Route path='/history' element={<ProtectedRoute><History /></ProtectedRoute>} />
                <Route path='/profile' element={<ProtectedRoute><Profile /></ProtectedRoute>} />
                <Route path='/chat' element={<ProtectedRoute><Chat /></ProtectedRoute>} />
                <Route path='/settings' element={<ProtectedRoute><Settings /></ProtectedRoute>} />
              </Routes>
            </AnimatePresence>
          </main>
        </div>
      </div>
      <Toaster position='top-right' toastOptions={{ 
        duration: 4000,
        style: { background: '#1e293b', color: '#fff', borderRadius: '12px' }
      }} />
    </BrowserRouter>
  );
}

export default App;
