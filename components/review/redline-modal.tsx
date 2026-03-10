"use client";

import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ArrowRight, Check } from "lucide-react";

interface RedlineModalProps {
  isOpen: boolean;
  onClose: () => void;
  originalText: string;
  suggestedText: string;
  onAccept: () => void;
}

export function RedlineModal({ isOpen, onClose, originalText, suggestedText, onAccept }: RedlineModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Compare AI Suggestion</DialogTitle>
          <DialogDescription>
            Review the proposed change before applying it to the contract.
          </DialogDescription>
        </DialogHeader>
        
        <div className="flex-1 grid grid-cols-2 gap-6 py-6 overflow-hidden">
          <div className="flex flex-col space-y-2 overflow-hidden">
            <span className="text-xs font-bold uppercase text-red-600 bg-red-50 px-2 py-1 rounded w-fit">Original</span>
            <div className="flex-1 p-4 border rounded-md bg-muted/20 overflow-auto whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground line-through decoration-red-300">
              {originalText}
            </div>
          </div>
          
          <div className="flex flex-col space-y-2 overflow-hidden">
            <span className="text-xs font-bold uppercase text-green-600 bg-green-50 px-2 py-1 rounded w-fit">AI Suggestion</span>
            <div className="flex-1 p-4 border border-green-200 rounded-md bg-green-50/30 overflow-auto whitespace-pre-wrap text-sm leading-relaxed">
              {suggestedText}
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="ghost" onClick={onClose}>
            Keep Original
          </Button>
          <Button onClick={onAccept} className="bg-green-600 hover:bg-green-700 text-white">
            <Check className="h-4 w-4 mr-2" />
            Accept & Replace
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
