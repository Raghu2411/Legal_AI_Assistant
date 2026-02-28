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
          cookiesToSet.forEach(({ name, value, options }) => request.cookies.set({ name, value, ...options }))
          supabaseResponse = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set({ name, value, ...options })
          )
        },
      },
    }
  )

  // IMPORTANT: Avoid calling getUser() multiple times in one request if possible.
  // We wrap in try-catch because if a refresh token is invalid/missing, 
  // getUser() might throw or log an error we want to handle gracefully.
  let user = null
  try {
    const { data, error } = await supabase.auth.getUser()
    if (error) {
      // Log error but don't crash
      console.warn('Middleware Auth User error:', error.message)
    }
    user = data?.user
  } catch (error) {
    // If we hit an auth error, we treat the user as logged out
    console.error('Middleware Auth Error:', error)
  }

  const url = new URL(request.url)
  const path = url.pathname

  // Public paths
  if (path === '/login' || path.startsWith('/auth/')) {
    if (user && path === '/login') {
      // If logged in, redirect away from login page
      return await redirectBasedOnRole(supabase, user.id, request, supabaseResponse)
    }
    return supabaseResponse
  }

  // Protected paths logic
  if (!user) {
    const loginUrl = new URL('/login', request.url)
    // Create a new redirect response but copy the updated cookies from supabaseResponse
    const redirectResponse = NextResponse.redirect(loginUrl)
    supabaseResponse.cookies.getAll().forEach(cookie => {
      redirectResponse.cookies.set(cookie)
    })
    return redirectResponse
  }

  // Role-based protection
  return await redirectBasedOnRole(supabase, user.id, request, supabaseResponse)
}

async function redirectBasedOnRole(supabase: any, userId: string, request: NextRequest, supabaseResponse: NextResponse) {
  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .single()

  const url = new URL(request.url)
  const path = url.pathname

  if (!profile) {
    // Missing profile edge case
    if (path !== '/access-denied') {
      const accessDeniedUrl = new URL('/access-denied', request.url)
      const redirectResponse = NextResponse.redirect(accessDeniedUrl)
      supabaseResponse.cookies.getAll().forEach(cookie => {
        redirectResponse.cookies.set(cookie)
      })
      return redirectResponse
    }
    return supabaseResponse
  }

  const role = profile.role

  // Admin access
  if (path.startsWith('/admin')) {
    if (role !== 'admin') {
      const dashboardUrl = new URL('/dashboard', request.url)
      const redirectResponse = NextResponse.redirect(dashboardUrl)
      supabaseResponse.cookies.getAll().forEach(cookie => {
        redirectResponse.cookies.set(cookie)
      })
      return redirectResponse
    }
  }

  // Lawyer access / Default redirects
  if (path === '/' || path === '/login') {
    const targetPath = role === 'admin' ? '/admin' : '/dashboard'
    const redirectResponse = NextResponse.redirect(new URL(targetPath, request.url))
    supabaseResponse.cookies.getAll().forEach(cookie => {
      redirectResponse.cookies.set(cookie)
    })
    return redirectResponse
  }

  return supabaseResponse
}
