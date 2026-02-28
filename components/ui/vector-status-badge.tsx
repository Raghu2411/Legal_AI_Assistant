import { Badge } from "@/components/ui/badge"

export type VectorStatus = 'Pending' | 'Processing' | 'Ready' | 'Error'

export function VectorStatusBadge({ status }: { status?: VectorStatus | string | null }) {
  if (!status) return null;

  switch (status) {
    case 'Pending':
      return <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">🟡 Pending</Badge>
    case 'Processing':
      return <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 animate-pulse">🔵 Vectorizing...</Badge>
    case 'Ready':
      return <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">🟢 Ready</Badge>
    case 'Error':
      return <Badge variant="destructive">🔴 Error</Badge>
    default:
      return null
  }
}
