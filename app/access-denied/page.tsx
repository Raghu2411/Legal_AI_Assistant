import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { signOut } from "@/app/auth/actions"

export default function AccessDeniedPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-muted/50 p-4">
      <Card className="max-w-md">
        <CardHeader>
          <CardTitle className="text-destructive">Access Denied</CardTitle>
          <CardDescription>
            You do not have a valid profile or sufficient permissions to access this area.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            If you believe this is an error, please contact your administrator to set up your profile role.
          </p>
        </CardContent>
        <CardFooter className="flex justify-between gap-4">
            <form action={signOut}>
                <Button variant="outline">Sign Out</Button>
            </form>
            <Button asChild>
                <Link href="/login">Back to Login</Link>
            </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
