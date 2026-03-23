"use client";

import { useEffect, useState, useCallback } from "react";
import { createClient } from "@/lib/supabase/client";
import { ObligationItem } from "./obligation-item";
import { ComplianceSidebar } from "./compliance-sidebar";

export function VerificationList({ clientId }: { clientId?: string }) {
  const [obligations, setObligations] = useState<any[]>([]);
  const [selectedOb, setSelectedOb] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();

  const fetchObligations = useCallback(async () => {
    let query = supabase
      .from("obligations")
      .select("*")
      .eq("status", "pending")
      .order("created_at", { ascending: false });

    if (clientId) {
      query = query.eq("client_id", clientId);
    }

    const { data } = await query;
    
    const results = data || [];
    setObligations(results);
    if (results.length > 0 && !selectedOb) {
      setSelectedOb(results[0]);
    }
    setLoading(false);
  }, [supabase, clientId, selectedOb]);

  useEffect(() => {
    fetchObligations();
    setSelectedOb(null); // Reset selection when filter changes

    const channel = supabase
      .channel("obligations_changes")
      .on("postgres_changes", { event: "*", schema: "public", table: "obligations" }, () => {
        fetchObligations();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [supabase, fetchObligations]);

  const handleSuccess = (id: string) => {
    setObligations(prev => {
      const filtered = prev.filter(ob => ob.id !== id);
      if (selectedOb?.id === id) {
        setSelectedOb(filtered.length > 0 ? filtered[0] : null);
      }
      return filtered;
    });
  };

  if (loading) {
    return <div className="text-center py-10">Loading pending obligations...</div>;
  }

  if (obligations.length === 0) {
    return (
      <div className="text-center py-10 text-muted-foreground border-2 border-dashed rounded-lg">
        No pending obligations to verify.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[600px]">
      <div className="lg:col-span-2 space-y-4 overflow-y-auto pr-2">
        {obligations.map((ob) => (
          <div key={ob.id} onClick={() => setSelectedOb(ob)} className={`cursor-pointer transition-opacity ${selectedOb?.id === ob.id ? 'opacity-100 ring-2 ring-primary rounded-lg' : 'opacity-80 hover:opacity-100'}`}>
            <ObligationItem obligation={ob} onSuccess={() => handleSuccess(ob.id)} />
          </div>
        ))}
      </div>
      <div className="hidden lg:block border-l pl-6">
        <ComplianceSidebar metadata={selectedOb?.metadata} />
      </div>
    </div>
  );
}
