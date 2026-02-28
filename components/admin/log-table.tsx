import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface Log {
  id: string
  event_type: string
  description: string
  created_at: string
  user_id: string
  profiles: {
    full_name: string
  } | null
}

export function LogTable({ logs }: { logs: Log[] }) {
  const getEventColor = (type: string) => {
    switch (type) {
      case 'LOGIN': return 'bg-green-500/10 text-green-500 hover:bg-green-500/20'
      case 'USER_DELETE': return 'bg-red-500/10 text-red-500 hover:bg-red-500/20'
      case 'ROLE_UPDATE': return 'bg-blue-500/10 text-blue-500 hover:bg-blue-500/20'
      case 'PLAYBOOK_UPLOAD': return 'bg-purple-500/10 text-purple-500 hover:bg-purple-500/20'
      default: return 'bg-slate-500/10 text-slate-500 hover:bg-slate-500/20'
    }
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Timestamp</TableHead>
            <TableHead>User</TableHead>
            <TableHead>Event</TableHead>
            <TableHead>Description</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {logs.map((log) => (
            <TableRow key={log.id}>
              <TableCell className="whitespace-nowrap text-muted-foreground">
                {new Date(log.created_at).toLocaleString()}
              </TableCell>
              <TableCell className="font-medium">
                {log.profiles?.full_name || 'System'}
              </TableCell>
              <TableCell>
                <div className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${getEventColor(log.event_type)}`}>
                  {log.event_type}
                </div>
              </TableCell>
              <TableCell>{log.description}</TableCell>
            </TableRow>
          ))}
          {logs.length === 0 && (
            <TableRow>
              <TableCell colSpan={4} className="h-24 text-center">
                No activity logs found.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
