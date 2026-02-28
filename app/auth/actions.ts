"use server"

import { createClient } from '@/lib/supabase/server'
import { logEvent } from '@/lib/supabase/admin'
import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'

export async function login(formData: FormData) {
  const supabase = createClient()

  const email = formData.get('email') as string
  const password = formData.get('password') as string

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })

  if (error) {
    redirect('/login?error=Could not authenticate user')
  }

  if (data.user) {
    await logEvent(data.user.id, 'LOGIN', `User ${email} logged in`)
  }

  revalidatePath('/', 'layout')
  redirect('/')
}

export async function signup(formData: FormData) {
  const supabase = createClient()

  const email = formData.get('email') as string
  const password = formData.get('password') as string
  const fullName = formData.get('full_name') as string

  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        full_name: fullName,
      }
    }
  })

  if (error) {
    redirect('/login?error=Could not register user')
  }

  if (data.user) {
    // 1. Create the profile with default 'lawyer' role
    // This satisfies Constitution Principle IV
    // Note: Inserting full_name as the user says email column doesn't exist
    const { error: profileError } = await supabase
      .from('profiles')
      .insert({
        id: data.user.id,
        full_name: fullName,
        role: 'lawyer', // Default role
      })

    if (profileError) {
      console.error('Error creating profile:', profileError)
    }

    await logEvent(data.user.id, 'USER_SIGNUP', `User ${fullName} (${email}) registered as lawyer`)
  }

  revalidatePath('/', 'layout')
  redirect('/')
}

export async function signOut() {
  const supabase = createClient()
  
  // Safely get user for logging before signing out
  let user = null
  try {
    const { data, error } = await supabase.auth.getUser()
    if (!error) {
      user = data?.user
    }
  } catch (error) {
    // If getUser fails, the user is likely already signed out or session is invalid
  }
  
  if (user) {
    await logEvent(user.id, 'LOGOUT', `User ${user.email} logged out`)
  }

  await supabase.auth.signOut()
  revalidatePath('/', 'layout')
  redirect('/login')
}
