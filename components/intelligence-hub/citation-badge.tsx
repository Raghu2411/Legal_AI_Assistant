import React from 'react';
import { Badge } from '@/components/ui/badge';

interface CitationBadgeProps {
  index: number;
  snippet: string;
  sourceName: string;
  onClick?: () => void;
}

export const CitationBadge: React.FC<CitationBadgeProps> = ({
  index,
  snippet,
  sourceName,
  onClick
}) => {
  return (
    <Badge
      variant="outline"
      className="inline-flex items-center justify-center h-4 px-1 text-[10px] font-bold cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors align-top ml-0.5 select-none"
      onClick={onClick}
      title={`${sourceName}: "${snippet.substring(0, 100)}..."`}
    >
      {index}
    </Badge>
  );
};
