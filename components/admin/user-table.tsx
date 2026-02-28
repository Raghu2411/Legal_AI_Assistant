"use client"

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Shield, ShieldAlert, Trash2, UserCog } from "lucide-react"
import { toggleUserRole, deleteUser } from "@/app/(admin)/admin/users/actions"
import { useState } from "react"

interface Profile {
  id: string
  full_name: string
  role: string
  created_at: string
}

export function UserTable({ initialUsers }: { initialUsers: Profile[] }) {
  const [users, setUsers] = useState(initialUsers)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const handleToggleRole = async (userId: string, currentRole: string) => {
    setLoadingId(userId)
    const result = await toggleUserRole(userId, currentRole)
    if (result.success) {
      setUsers(users.map(u => 
        u.id === userId 
          ? { ...u, role: currentRole === 'admin' ? 'lawyer' : 'admin' } 
          : u
      ))
    } else if (result.error) {
      alert(`Error updating role: ${result.error}`)
    }
    setLoadingId(null)
  }

  const handleDelete = async (userId: string) => {
    if (!confirm("Are you sure? This lawyer's data will be reassigned to you.")) return
    
    setLoadingId(userId)
    const result = await deleteUser(userId)
    if (result.success) {
      setUsers(users.filter(u => u.id !== userId))
    } else if (result.error) {
      alert(`Error deleting user: ${result.error}`)
    }
    setLoadingId(null)
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>User Name</TableHead>
            <TableHead>Role</TableHead>
            <TableHead>Joined</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {users.map((user) => (
            <TableRow key={user.id}>
              <TableCell className="font-medium">
                {user.full_name || "Unknown User"}
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  {user.role === 'admin' ? (
                    <ShieldAlert className="h-4 w-4 text-destructive" />
                  ) : (
                    <Shield className="h-4 w-4 text-primary" />
                  )}
                  <span className="capitalize">{user.role}</span>
                </div>
              </TableCell>
              <TableCell>{new Date(user.created_at).toLocaleDateString()}</TableCell>
              <TableCell className="text-right">
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={loadingId === user.id}
                    onClick={() => handleToggleRole(user.id, user.role)}
                  >
                    <UserCog className="h-4 w-4 mr-2" />
                    Toggle Role
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={loadingId === user.id}
                    onClick={() => handleDelete(user.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
          {users.length === 0 && (
            <TableRow>
              <TableCell colSpan={4} className="h-24 text-center">
                No users found.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
