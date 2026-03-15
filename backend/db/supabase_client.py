import os
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """
    Initializes and returns a Supabase client.
    Requires SUPABASE_URL and SUPABASE_KEY environment variables.
    """
    url: str = os.environ.get("SUPABASE_URL", "https://xyzcompany.supabase.co")
    key: str = os.environ.get("SUPABASE_KEY", "public-anon-key")

    # Supabase Client initialization
    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")
        return None

# Singleton-like instance
supabase_client = get_supabase_client()
