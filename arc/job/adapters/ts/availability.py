"""
The availability check of TS search adapters.

Many TS search adapters drive an external engine (a conda env, an executable, or an ESS server).
The Scheduler asks :func:`ts_adapter_unavailable_reason` about each adapter it is about to spawn
for a reaction, and drops an adapter whose engine is not installed or not configured, rather than
let it raise, stall, or run as an empty job. Each engine rule lives once, in its adapter's module
(``engine_unavailable_reason()``), where the adapter's own execute-time guard also uses it.
"""

from arc.imports import settings
from arc.job.adapters.ts import autotst_ts, gcn_ts, goflow_ts, kinbot_ts, rits_ts, xtb_gsm
from arc.job.factory import _registered_job_adapters

global_ess_settings, servers = settings['global_ess_settings'], settings['servers']

ENGINE_FREE_TS_ADAPTERS = ('heuristics', 'linear')

ENGINE_REASON_FUNCTIONS = {'autotst': autotst_ts.engine_unavailable_reason,
                           'gcn': gcn_ts.engine_unavailable_reason,
                           'goflow': goflow_ts.engine_unavailable_reason,
                           'kinbot': kinbot_ts.engine_unavailable_reason,
                           'rits': rits_ts.engine_unavailable_reason,
                           'xtb_gsm': xtb_gsm.engine_unavailable_reason,
                           }


def orca_neb_unavailable_reason(ess_settings: dict | None) -> str | None:
    """
    Report why Orca NEB cannot run. It is a queue adapter that runs Orca, and it resolves its server
    from the ESS settings' 'orca' entry (``OrcaAdapter.__init__`` resolves the server while the job
    adapter is still 'orca'), so it needs a server there, defined in the servers settings.
    Empty ESS settings fall back to ``global_ess_settings``, as the job itself does.

    Args:
        ess_settings (dict, optional): The run's ESS settings, mapping software to a server or a list of servers.

    Returns:
        str | None: ``None`` if Orca NEB has a usable server, otherwise the reason.
    """
    server_list = (ess_settings or global_ess_settings or dict()).get('orca')
    server_list = [server_list] if isinstance(server_list, str) else server_list
    if not isinstance(server_list, list) or not server_list:
        return (f"no server is set for 'orca' in the ESS settings (got {server_list!r}), and Orca NEB "
                f"runs on the Orca server; map 'orca' to a server that runs Orca in global_ess_settings")
    unknown = [server for server in server_list if not isinstance(server, str) or server not in servers]
    if unknown:
        return f"the 'orca' ESS settings name servers {unknown} that are not defined in the servers settings"
    return None


SERVER_REASON_FUNCTIONS = {'orca_neb': orca_neb_unavailable_reason}


def ts_adapter_unavailable_reason(adapter: str, ess_settings: dict | None) -> str | None:
    """
    Report why a TS search adapter cannot run on this machine with these settings.

    Args:
        adapter (str): The TS adapter name, e.g., ``'kinbot'``.
        ess_settings (dict, optional): The run's ESS settings.

    Returns:
        str | None: ``None`` if the adapter can run, otherwise one sentence naming the missing
                    env, executable, or setting.
    """
    adapter = adapter.lower()
    if adapter not in _registered_job_adapters:
        return f'no {adapter!r} adapter is registered in this ARC build, so it has no module to run'
    if adapter in ENGINE_FREE_TS_ADAPTERS or 'mock' in adapter:
        return None
    if adapter in SERVER_REASON_FUNCTIONS:
        return SERVER_REASON_FUNCTIONS[adapter](ess_settings)
    reason_function = ENGINE_REASON_FUNCTIONS.get(adapter)
    return reason_function() if reason_function is not None else None
