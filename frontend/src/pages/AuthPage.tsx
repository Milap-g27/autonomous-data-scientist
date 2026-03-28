import { AuthUI } from "../components/ui/auth-fuse";
import { Navigate } from "react-router-dom";
import { useAuthContext } from "@/context/AuthContext";

export default function AuthPage() {
  const { user, loading } = useAuthContext();

  if (!loading && user) {
    return <Navigate to="/" replace />;
  }

  return <AuthUI />;
}
