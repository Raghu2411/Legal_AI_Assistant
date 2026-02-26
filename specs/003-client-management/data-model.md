# Data Model: Client & Case Management

## Database Schema (PostgreSQL)

### Table: `clients`
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key (Default: uuid_generate_v4()) |
| auto_case_id | text | Unique firm-wide ID (Generated via trigger) |
| name | text | Client name (Required) |
| case_type | text | Type of legal case (Required) |
| lawyer_id | uuid | Foreign Key (references profiles.id, Owner) |
| status | text | 'Active', 'Closed', 'Archived' (Default: 'Active') |
| created_at | timestamp | Creation timestamp |

### Table: `documents`
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key |
| client_id | uuid | Foreign Key (references clients.id) |
| file_url | text | Path in `client-vaults` storage bucket |
| file_name | text | Original file name for display |
| doc_type | text | 'Contract', 'Evidence', 'Correspondence', 'Pleading' |
| uploaded_by | uuid | Foreign Key (references profiles.id) |
| uploaded_at | timestamp | Upload timestamp |

## Automated Case ID Logic

### Trigger Function: `generate_client_case_id()`
```sql
CREATE OR REPLACE FUNCTION generate_client_case_id()
RETURNS TRIGGER AS $$
DECLARE
    lawyer_name TEXT;
    name_slug TEXT;
    random_suffix TEXT;
    final_id TEXT;
    done BOOLEAN := FALSE;
BEGIN
    -- Fetch lawyer's name
    SELECT full_name INTO lawyer_name FROM profiles WHERE id = NEW.lawyer_id;
    
    -- Extract first word and clean it
    name_slug := lower(split_part(lawyer_name, ' ', 1));
    name_slug := regexp_replace(name_slug, '[^a-z0-9]', '', 'g');

    -- Generate and check for uniqueness
    WHILE NOT done LOOP
        random_suffix := upper(substring(replace(gen_random_uuid()::text, '-', ''), 1, 4));
        final_id := name_slug || '-' || random_suffix;
        
        IF NOT EXISTS (SELECT 1 FROM clients WHERE auto_case_id = final_id) THEN
            done := TRUE;
        END IF;
    END LOOP;

    NEW.auto_case_id := final_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER trigger_generate_case_id
BEFORE INSERT ON clients
FOR EACH ROW
EXECUTE FUNCTION generate_client_case_id();
```

## Row-Level Security (RLS)

### `clients` Policies
```sql
ALTER TABLE clients ENABLE ROW LEVEL SECURITY;

-- Select: Owner or Admin
CREATE POLICY "Lawyers can view their own clients" ON clients
FOR SELECT USING (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

-- Insert: Owner or Admin
CREATE POLICY "Lawyers can insert their own clients" ON clients
FOR INSERT WITH CHECK (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

-- Update: Owner or Admin
CREATE POLICY "Lawyers can update their own clients" ON clients
FOR UPDATE USING (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

-- Delete: Admin only
CREATE POLICY "Only admins can delete clients" ON clients
FOR DELETE USING ((SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
```

### `documents` Policies
```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Select: If user has access to the client
CREATE POLICY "Access via client ownership" ON documents
FOR SELECT USING (
    EXISTS (
        SELECT 1 FROM clients 
        WHERE id = documents.client_id 
        AND (lawyer_id = auth.uid() OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin')
    )
);

-- Delete: Uploader or Admin
CREATE POLICY "Uploader or admin can delete documents" ON documents
FOR DELETE USING (uploaded_by = auth.uid() OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
```

## Storage Policies

### Bucket: `client-vaults`
- **Read**: `(auth.uid() = lawyer_id) OR (is_admin)` (Requires join with `clients` table in bucket policy).
- **Write**: Authenticated users can upload to their client's folder.
