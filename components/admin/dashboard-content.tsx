"use client"

import { useEffect, useState } from "react"
import { createClient } from "@/lib/supabase/client"
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { 
  Users, 
  History, 
  BookOpen, 
  TrendingUp 
} from "lucide-react"
import { LogTable } from "@/components/admin/log-table"

interface DashboardProps {
  initialUserCount: number
  initialLogs: any[]
  userEmail: string
  currentVersion: number
  recentLogCount: number
}

export function DashboardContent({ 
  initialUserCount, 
  initialLogs, 
  userEmail, 
  currentVersion,
  recentLogCount
}: DashboardProps) {
  const [userCount, setUserCount] = useState(initialUserCount)
  const [logs, setLogs] = useState(initialLogs)
  const [recentEvents, setRecentEvents] = useState(recentLogCount)
  const supabase = createClient()

  useEffect(() => {
    // ... rest of useEffect ...
    const profileChannel = supabase
      .channel('profile-changes')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'profiles' },
        async () => {
          const { count } = await supabase
            .from('profiles')
            .select('*', { count: 'exact', head: true })
          if (count !== null) setUserCount(count)
        }
      )
      .subscribe()

    // 2. Subscribe to logs for recent activity
    const logChannel = supabase
      .channel('log-changes')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'logs' },
        async (payload) => {
          // Fetch the log and profile manually
          const { data: log } = await supabase
            .from('logs')
            .select('*')
            .eq('id', payload.new.id)
            .single()

          if (log) {
            const { data: profile } = await supabase
              .from('profiles')
              .select('full_name')
              .eq('id', log.user_id)
              .single()

            const newLog = {
              ...log,
              profiles: {
                full_name: profile?.full_name || 'System'
              }
            }
            setLogs(prev => [newLog, ...prev].slice(0, 5))
          }
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(profileChannel)
      supabase.removeChannel(logChannel)
    }
  }, [supabase])

  const stats = [
    {
      title: "Total Users",
      value: userCount,
      icon: Users,
      description: "Registered lawyers and admins",
    },
    {
      title: "Active Playbook",
      value: currentVersion > 0 ? `v${currentVersion}` : "None",
      icon: BookOpen,
      description: "Current firm guidelines",
    },
    {
      title: "System Events",
      value: recentEvents,
      icon: History,
      description: "Last 24 hours activity",
    },
    {
      title: "AI Inquiries",
      value: "24h",
      icon: TrendingUp,
      description: "Usage trend: Stable",
    },
  ]

  return (
    <div className="p-8 flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Admin Overview</h1>
        <p className="text-muted-foreground">Welcome back, {userEmail}.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.title}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                {stat.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-1">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <LogTable logs={logs} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
