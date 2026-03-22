"use client";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useRouter, useSearchParams } from "next/navigation";

interface Client {
  id: string;
  name: string;
}

export function ClientFilter({ clients }: { clients: Client[] }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedClient = searchParams.get("clientId") || "all";

  const handleClientChange = (value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (value === "all") {
      params.delete("clientId");
    } else {
      params.set("clientId", value);
    }
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-sm font-medium">Filter by Client:</span>
      <Select value={selectedClient} onValueChange={handleClientChange}>
        <SelectTrigger className="w-[280px]">
          <SelectValue placeholder="Select a client" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Clients</SelectItem>
          {clients.map((client) => (
            <SelectItem key={client.id} value={client.id}>
              {client.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
