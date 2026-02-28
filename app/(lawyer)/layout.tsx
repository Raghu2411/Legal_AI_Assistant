import Link from "next/link";
import { Button } from "@/components/ui/button";
import { signOut } from "@/app/auth/actions";
import { 
  Users, 
  LayoutDashboard,
  LogOut,
  Briefcase
} from "lucide-react";

export default function LawyerLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-64 border-r bg-muted/30 p-4 flex flex-col gap-6">
        <div className="flex items-center gap-2 px-2">
          <Briefcase className="h-6 w-6 text-primary" />
          <span className="text-xl font-bold tracking-tight">SAI-Legal Lawyer</span>
        </div>
        
        <nav className="flex-1 flex flex-col gap-1">
          <Link href="/dashboard">
            <span className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent transition-colors">
              <LayoutDashboard className="h-4 w-4" />
              Dashboard
            </span>
          </Link>
          <Link href="/clients">
            <span className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent transition-colors">
              <Users className="h-4 w-4" />
              Client Portfolio
            </span>
          </Link>
        </nav>

        <div className="border-t pt-4">
          <form action={signOut}>
            <Button variant="ghost" className="w-full justify-start gap-3 px-3">
              <LogOut className="h-4 w-4" />
              Sign Out
            </Button>
          </form>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {children}
      </main>
    </div>
  );
}
