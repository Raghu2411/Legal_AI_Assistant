import React from 'react';
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { RiskStatus } from "@/lib/review/schemas";
import { cn } from "@/lib/utils";
import { AlertCircle, AlertTriangle, CheckCircle2, ArrowRight, Ghost } from "lucide-react";
import { Button } from "@/components/ui/button";

interface RiskItemProps {
  status: RiskStatus;
  originalText: string;
  rationale: string;
  suggestedRewrite?: string;
  isGap?: boolean;
  onClick?: () => void;
  onViewSuggestion?: () => void;
  isSelected?: boolean;
}

export function RiskItem({ 
  status, 
  originalText, 
  rationale, 
  suggestedRewrite,
  isGap,
  onClick, 
  onViewSuggestion,
  isSelected 
}: RiskItemProps) {
  const statusConfig = {
    red: {
      color: "bg-red-50 text-red-700 border-red-200",
      icon: isGap ? <Ghost className="h-4 w-4 text-red-600" /> : <AlertCircle className="h-4 w-4 text-red-600" />,
      badge: isGap ? "Missing Clause" : "High Risk",
    },
    yellow: {
      color: "bg-yellow-50 text-yellow-700 border-yellow-200",
      icon: <AlertTriangle className="h-4 w-4 text-yellow-600" />,
      badge: "Medium Risk",
    },
    green: {
      color: "bg-green-50 text-green-700 border-green-200",
      icon: <CheckCircle2 className="h-4 w-4 text-green-600" />,
      badge: "Compliant",
    },
  };

  const config = statusConfig[status];

  return (
    <Card 
      className={cn(
        "p-3 cursor-pointer transition-all hover:ring-2 hover:ring-primary/20",
        config.color,
        isSelected && "ring-2 ring-primary shadow-md",
        isGap && "border-dashed border-2"
      )}
      onClick={onClick}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          {config.icon}
          <Badge variant="outline" className={cn("text-[10px] font-bold uppercase", config.color)}>
            {config.badge}
          </Badge>
          {isGap && (
            <Badge className="bg-red-600 text-white text-[10px] font-bold uppercase border-none animate-pulse">
              Gap
            </Badge>
          )}
        </div>
        {suggestedRewrite && (
          <Button 
            variant="ghost" 
            size="sm" 
            className="h-6 px-2 text-[10px] uppercase font-bold text-primary hover:bg-primary/10"
            onClick={(e) => {
              e.stopPropagation();
              onViewSuggestion?.();
            }}
          >
            {isGap ? "Insert Clause" : "Review AI Suggestion"}
            <ArrowRight className="ml-1 h-3 w-3" />
          </Button>
        )}
      </div>
      <div className="space-y-2">
        <p className={cn(
          "text-xs font-medium line-clamp-2 italic border-l-2 border-current pl-2",
          isGap ? "text-red-900" : "text-muted-foreground"
        )}>
          "{originalText}"
        </p>
        <p className="text-xs">
          {rationale}
        </p>
      </div>
    </Card>
  );
}
