import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function updateSession(request: NextRequest) {
  let supabaseResponse = NextResponse.next({
    request,
  })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) => request.cookies.set(name, value, options))
          supabaseResponse = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  const { data: { user } } = await supabase.auth.getUser()

  const url = new URL(request.url)
  const path = url.pathname

  // Public paths
  if (path === '/login' || path.startsWith('/auth/')) {
    if (user && path === '/login') {
      // If logged in, redirect away from login page
      return await redirectBasedOnRole(supabase, user.id, request)
    }
    return supabaseResponse
  }

  // Protected paths logic
  if (!user) {
    const loginUrl = new URL('/login', request.url)
    return NextResponse.redirect(loginUrl)
  }

  // Role-based protection
  return await redirectBasedOnRole(supabase, user.id, request)
}

async function redirectBasedOnRole(supabase: any, userId: string, request: NextRequest) {
  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .single()

  const url = new URL(request.url)
  const path = url.pathname

  if (!profile) {
    // Missing profile edge case (User Story 2 / Clarification)
    if (path !== '/access-denied') {
        return NextResponse.redirect(new URL('/access-denied', request.url))
    }
    return NextResponse.next()
  }

  const role = profile.role

  // Admin access
  if (path.startsWith('/admin')) {
    if (role !== 'admin') {
      return NextResponse.redirect(new URL('/dashboard', request.url))
    }
  }

  // Lawyer access (Admins can access dashboard too per spec 2.2)
  if (path === '/login' || path === '/') {
    if (role === 'admin') {
        return NextResponse.redirect(new URL('/admin', request.url))
    } else {
        return NextResponse.redirect(new URL('/dashboard', request.url))
    }
  }

  return NextResponse.next()
}
