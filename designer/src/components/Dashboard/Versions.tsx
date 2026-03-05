import { Navigate } from 'react-router-dom'

/** Legacy Versions page — redirects to Deploy. */
const Versions = () => <Navigate to="/chat/deploy" replace />

export default Versions
