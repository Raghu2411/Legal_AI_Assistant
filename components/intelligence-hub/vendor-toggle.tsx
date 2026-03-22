'use client';

import { Label } from '@/components/ui/label';
import { ShieldCheck, ShieldAlert } from 'lucide-react';
import { cn } from '@/lib/utils';

interface VendorToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export function VendorToggle({ enabled, onToggle }: VendorToggleProps) {
  return (
    <div 
      className={cn(
        "flex items-center justify-between p-4 rounded-lg border transition-all cursor-pointer select-none",
        enabled 
          ? "bg-primary/5 border-primary shadow-sm" 
          : "bg-muted/20 border-transparent"
      )}
      onClick={() => onToggle(!enabled)}
    >
      <div className="flex items-center gap-3">
        <div className={cn(
          "p-2 rounded-full",
          enabled ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
        )}>
          {enabled ? <ShieldCheck className="h-5 w-5" /> : <ShieldAlert className="h-5 w-5" />}
        </div>
        <div>
          <Label className="text-base font-bold cursor-pointer">Vendor Mode</Label>
          <p className="text-xs text-muted-foreground">
            {enabled 
              ? "Isolating retrieval to vendor-related procurement documents only." 
              : "Retrieval is searching the entire client document vault."
            }
          </p>
        </div>
      </div>
      
      <div className={cn(
        "relative w-12 h-6 rounded-full transition-colors",
        enabled ? "bg-primary" : "bg-muted-foreground/30"
      )}>
        <div className={cn(
          "absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform",
          enabled ? "translate-x-6" : "translate-x-0"
        )} />
      </div>
    </div>
  );
}
