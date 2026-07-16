import { NavLink } from 'react-router-dom';

const LINKS = [
  { to: '/', label: 'Run configuration', end: true },
  { to: '/results', label: 'Results' },
  { to: '/compare', label: 'Model comparison' },
  { to: '/health', label: 'System health' },
];

export default function Nav() {
  return (
    <nav className="app-nav">
      <span className="brand">LLM Eval Harness</span>
      {LINKS.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.end}
          className={({ isActive }) => (isActive ? 'active' : '')}
        >
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
