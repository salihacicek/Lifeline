import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, User, Settings, FileText } from 'lucide-react';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <aside className="sidebar glass-panel">
      <div className="sidebar-header">
        <Activity className="logo-icon text-gradient" size={32} />
        <h2 className="logo-text text-gradient">ADS1293 AI</h2>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Activity size={20} />
          <span>Monitor</span>
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <FileText size={20} />
          <span>Geçmiş Kayıtlar</span>
        </NavLink>
        <NavLink to="/patients" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <User size={20} />
          <span>Hastalar</span>
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Settings size={20} />
          <span>Ayarlar</span>
        </NavLink>
      </nav>

        <div className="status-indicator" style={{marginTop: 'auto'}}>
          <div className="pulse-dot"></div>
          <span className="indicator-text">Sistem Aktif</span>
        </div>
    </aside>
  );
};

export default Sidebar;
