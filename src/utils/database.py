"""
Database utility functions for connecting to PostgreSQL and executing queries
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import pandas as pd
from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.db_config = config['database']
        self._engine = None
        
    def get_connection_string(self, include_db: bool = True) -> str:
        """
        Get PostgreSQL connection string
        
        Args:
            include_db: Whether to include database name in connection string
            
        Returns:
            Connection string
        """
        db_name = self.db_config['name'] if include_db else 'postgres'
        return (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{db_name}"
        )
    
    def get_engine(self):
        """Get or create SQLAlchemy engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.get_connection_string(),
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection as a context manager
        
        Yields:
            psycopg2 connection
        """
        conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            database=self.db_config['name']
        )
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = False):
        """
        Get a database cursor as a context manager
        
        Args:
            dict_cursor: Whether to use RealDictCursor (returns rows as dicts)
            
        Yields:
            psycopg2 cursor
        """
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None, 
                     fetch: bool = False) -> Optional[List[Dict]]:
        """
        Execute a SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        with self.get_cursor(dict_cursor=fetch) as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None
    
    def execute_many(self, query: str, data: List[tuple], batch_size: int = 1000):
        """
        Execute a query with multiple parameter sets (batch insert)
        
        Args:
            query: SQL query string
            data: List of parameter tuples
            batch_size: Number of rows per batch
        """
        with self.get_cursor() as cursor:
            execute_batch(cursor, query, data, page_size=batch_size)
    
    def read_sql(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Read SQL query into a pandas DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        engine = self.get_engine()
        return pd.read_sql(query, engine, params=params)
    
    def to_sql(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append',
               index: bool = False, chunksize: int = 1000):
        """
        Write DataFrame to SQL table
        
        Args:
            df: DataFrame to write
            table_name: Target table name
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
            chunksize: Number of rows per batch
        """
        engine = self.get_engine()
        df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method='multi'
        )
        logger.info(f"Written {len(df)} rows to table '{table_name}'")
    
    def bulk_insert(self, table_name: str, df: pd.DataFrame, 
                   batch_size: int = 10000):
        """
        Bulk insert DataFrame into table using COPY
        
        Args:
            table_name: Target table name
            df: DataFrame to insert
            batch_size: Number of rows per batch
        """
        # Convert DataFrame to CSV in memory
        from io import StringIO
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Split into batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                # Create CSV buffer
                buffer = StringIO()
                batch.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                
                # Use COPY for fast insertion
                columns = ', '.join(batch.columns)
                cursor.copy_expert(
                    f"COPY {table_name} ({columns}) FROM STDIN WITH CSV",
                    buffer
                )
                
                logger.debug(f"Inserted batch {i//batch_size + 1} "
                           f"({len(batch)} rows) into {table_name}")
            
            logger.info(f"Bulk inserted {len(df)} rows into {table_name}")
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists
        
        Args:
            table_name: Table name to check
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """
        result = self.execute_query(query, (table_name,), fetch=True)
        return result[0]['exists'] if result else False
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get row count for a table
        
        Args:
            table_name: Table name
            
        Returns:
            Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        result = self.execute_query(query, fetch=True)
        return result[0]['count'] if result else 0
    
    def truncate_table(self, table_name: str):
        """
        Truncate a table (delete all rows)
        
        Args:
            table_name: Table name to truncate
        """
        query = f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;"
        self.execute_query(query)
        logger.info(f"Truncated table '{table_name}'")
    
    def get_latest_date(self, table_name: str, date_column: str = 'timestamp') -> Optional[str]:
        """
        Get the latest date from a table
        
        Args:
            table_name: Table name
            date_column: Name of the date/timestamp column
            
        Returns:
            Latest date as string, or None if table is empty
        """
        query = f"SELECT MAX({date_column}) as max_date FROM {table_name};"
        result = self.execute_query(query, fetch=True)
        
        if result and result[0]['max_date']:
            return result[0]['max_date'].strftime('%Y-%m-%d')
        return None
    
    def create_spatial_index(self, table_name: str, column_name: str = 'location'):
        """
        Create a spatial index on a geometry column
        
        Args:
            table_name: Table name
            column_name: Geometry column name
        """
        index_name = f"idx_{table_name}_{column_name}"
        query = f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {table_name} USING GIST({column_name});
        """
        self.execute_query(query)
        logger.info(f"Created spatial index '{index_name}'")
    
    def vacuum_analyze(self, table_name: Optional[str] = None):
        """
        Run VACUUM ANALYZE to optimize database
        
        Args:
            table_name: Specific table to vacuum, or None for entire database
        """
        with self.get_connection() as conn:
            conn.set_isolation_level(0)  # AUTOCOMMIT mode required for VACUUM
            cursor = conn.cursor()
            
            if table_name:
                cursor.execute(f"VACUUM ANALYZE {table_name};")
                logger.info(f"Vacuumed and analyzed table '{table_name}'")
            else:
                cursor.execute("VACUUM ANALYZE;")
                logger.info("Vacuumed and analyzed entire database")
            
            cursor.close()


# Singleton instance
_db_manager = None

def get_db_manager(config_path: Optional[str] = None) -> DatabaseManager:
    """
    Get or create DatabaseManager singleton instance
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config_path)
    return _db_manager
