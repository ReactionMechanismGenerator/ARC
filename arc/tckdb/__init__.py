"""ARC-side TCKDB integration: build, write, and optionally upload conformer/calculation payloads.

The chemistry/provenance mapping lives here in ARC. Transport lives in
``tckdb-client``. Server-side validation/persistence lives in TCKDB.
"""

from arc.tckdb.adapter import TCKDBAdapter, UploadOutcome
from arc.tckdb.config import TCKDBConfig

__all__ = ["TCKDBAdapter", "TCKDBConfig", "UploadOutcome"]
