"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CalendarItemPopover } from "./calendar-item-popover";

export function CalendarView({ clientId }: { clientId?: string }) {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [obligations, setObligations] = useState<any[]>([]);
  const [selectedOb, setSelectedOb] = useState<any>(null);
  const supabase = createClient();

  const fetchConfirmed = async () => {
    let query = supabase
      .from("obligations")
      .select(`
        *,
        clients (name, auto_case_id),
        documents (file_name)
      `)
      .eq("status", "confirmed");

    if (clientId) {
      query = query.eq("client_id", clientId);
    }

    const { data } = await query;
    setObligations(data || []);
  };

  useEffect(() => {
    fetchConfirmed();

    const channel = supabase
      .channel("confirmed_obligations")
      .on("postgres_changes", { event: "*", schema: "public", table: "obligations" }, () => {
        fetchConfirmed();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [supabase, clientId]);

  const handleActionSuccess = (id: string) => {
    setObligations(prev => prev.filter(ob => ob.id !== id));
    setSelectedOb(null);
  };

  const daysInMonth = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0).getDate();
  const firstDayOfMonth = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1).getDay();

  const days = Array.from({ length: daysInMonth }, (_, i) => i + 1);
  const blanks = Array.from({ length: firstDayOfMonth }, (_, i) => i);

  const prevMonth = () => setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
  const nextMonth = () => setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));

  const monthName = currentDate.toLocaleString('default', { month: 'long' });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">{monthName} {currentDate.getFullYear()}</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="icon" onClick={prevMonth}><ChevronLeft className="h-4 w-4" /></Button>
          <Button variant="outline" size="icon" onClick={nextMonth}><ChevronRight className="h-4 w-4" /></Button>
        </div>
      </div>
      <div className="grid grid-cols-7 gap-px bg-muted rounded-lg overflow-hidden border">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(d => (
          <div key={d} className="bg-background p-2 text-center text-xs font-medium text-muted-foreground">{d}</div>
        ))}
        {blanks.map(b => <div key={`b-${b}`} className="bg-background min-h-[80px] p-2" />)}
        {days.map(day => {
          const dateStr = `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
          const dayObligations = obligations.filter(ob => ob.due_date?.startsWith(dateStr));

          return (
            <div key={day} className="bg-background min-h-[80px] p-2 border-t border-l">
              <span className="text-xs text-muted-foreground">{day}</span>
              <div className="mt-1 space-y-1">
                {dayObligations.map(ob => (
                  <div 
                    key={ob.id} 
                    className="text-[10px] bg-blue-100 text-blue-700 p-1 rounded truncate cursor-pointer hover:bg-blue-200 transition-colors" 
                    title={`${ob.clients?.name}: ${ob.description}`}
                    onClick={() => setSelectedOb(ob)}
                  >
                    <span className="font-bold">[{ob.clients?.name}]</span> {ob.description}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      {selectedOb && (
        <CalendarItemPopover 
          obligation={selectedOb} 
          onClose={() => setSelectedOb(null)} 
          onSuccess={() => handleActionSuccess(selectedOb.id)}
        />
      )}
    </div>
  );
}
