import React from 'react';
import { FaGithub, FaTwitter, FaLinkedin } from 'react-icons/fa';

export default function Footer() {
  return (
    <footer className="bg-white border-t mt-12 py-6">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-500 text-sm">
            © 2026 Insomnia Risk Predictor. All rights reserved.
          </p>
          <div className="flex gap-4 mt-4 md:mt-0">
            <FaGithub className="text-gray-400 hover:text-primary-600 cursor-pointer text-xl" />
            <FaTwitter className="text-gray-400 hover:text-primary-600 cursor-pointer text-xl" />
            <FaLinkedin className="text-gray-400 hover:text-primary-600 cursor-pointer text-xl" />
          </div>
        </div>
      </div>
    </footer>
  );
}
