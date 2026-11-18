# Project memory

- At the start of a task, read `WEEKLY_LOG.md` once for recent context. Read archived logs only when relevant to the request.
- Before finishing substantive work, append one dated entry to `WEEKLY_LOG.md`: summary of changes, key decisions, validation performed, and open issues or next steps. Use 3–5 short bullets, ideally under 120 words. Skip entries for trivial questions or work already logged.
- Record only verified outcomes. Distinguish changes in the working tree from committed work; never infer completed work from existing modifications. Do not store secrets, raw command output, or chat transcripts.
- Keep overhead low: do not scan repository history, run tests, or launch agents solely to write a log entry. Use information already gathered during the task.
- Use dates in America/New_York. Keep the current week's entries in `WEEKLY_LOG.md`. On the first substantive task of a new week, move the previous week's entries to `archive/YYYY-Www.md` (ISO week), preserving unresolved issues in the new log. Never overwrite an existing archive; append without duplicating entries.
- When asked for a weekly presentation, combine the relevant logs with the last seven days of Git history, distinguish committed changes from local work, and identify any claims that need confirmation. Archive after generating the presentation if requested.
