"""Build a future add/modify-only update from an explicitly curated change directory.
Standard library only; never stages, commits, pushes, deletes, or runs experiment code.
"""
import argparse, csv, hashlib, io, json, os, subprocess, shutil
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--changes',required=True,type=Path,help='Curated files using repository-relative paths')
    p.add_argument('--out',required=True,type=Path,help='New directory; must not exist')
    p.add_argument('--id',required=True,help='e.g. 002 or W16')
    a=p.parse_args()
    repo,changes,out=a.repo.resolve(),a.changes.resolve(),a.out.resolve()
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args]).decode('utf-8').strip()
    if out.exists():raise SystemExit('Output exists; refusing to overwrite.')
    if not changes.is_dir():raise SystemExit('Changes directory is missing.')
    if changes==repo or changes in repo.parents or repo in changes.parents:raise SystemExit('Use a curated changes folder outside the Git repository.')
    if changes in out.parents or out in changes.parents:raise SystemExit('Output and changes folders must not contain each other.')
    if git('status','--porcelain'):raise SystemExit('Base repository must be clean.')
    if git('branch','--show-current')!='main':raise SystemExit('Base repository must be on main.')
    base=git('rev-parse','HEAD')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if base!=remote:raise SystemExit('Local main must equal remote main before building.')
    rows=[]; data={}; folded=set()
    for f in sorted(changes.rglob('*')):
        if f.is_symlink():raise SystemExit('Symlinks are not supported: '+str(f))
        if not f.is_file():continue
        rel=f.relative_to(changes).as_posix()
        if '.git' in f.relative_to(changes).parts or ':' in rel:raise SystemExit('Unsafe path: '+rel)
        if rel.casefold() in folded:raise SystemExit('Case-insensitive path collision: '+rel)
        folded.add(rel.casefold())
        new=f.read_bytes()
        if len(new)>=100*1024**2:raise SystemExit('File too large for ordinary GitHub upload: '+rel)
        target=repo/rel
        if target.is_symlink() or target.is_dir():raise SystemExit('Target is not a regular file: '+rel)
        old=target.read_bytes() if target.is_file() else None
        if new==old:continue
        try: old_lf=sha(old.decode('utf-8').replace('\r\n','\n').encode()) if old is not None else ''
        except UnicodeDecodeError: old_lf=''
        rows.append(dict(path=rel,action='MODIFY' if old is not None else 'ADD',bytes=len(new),sha256=sha(new),old_sha256=sha(old) if old is not None else '',old_lf_sha256=old_lf))
        data[rel]=new
    if not rows:raise SystemExit('No added or changed files.')
    # Preserve bytes only for the delta's paths, without changing checkout rules for unrelated files.
    attr_path=repo/'.gitattributes'
    old_attr=attr_path.read_bytes() if attr_path.is_file() else None
    attr=data.get('.gitattributes',old_attr or b'')
    attr+=b'\n# Exact bytes for incremental update '+a.id.encode()+b'\n'
    for name in sorted(set(data)|{'.gitattributes'}):
        pattern='/'+name.replace('*','\\*').replace('?','\\?').replace('[','\\[')
        attr+=(json.dumps(pattern,ensure_ascii=False)+' -text\n').encode()
    rows=[r for r in rows if r['path']!='.gitattributes']
    try: attr_lf=sha(old_attr.decode('utf-8').replace('\r\n','\n').encode()) if old_attr else ''
    except UnicodeDecodeError:attr_lf=''
    rows.append(dict(path='.gitattributes',action='MODIFY' if old_attr is not None else 'ADD',bytes=len(attr),sha256=sha(attr),old_sha256=sha(old_attr) if old_attr is not None else '',old_lf_sha256=attr_lf))
    data['.gitattributes']=attr
    b=io.StringIO(newline='');w=csv.DictWriter(b,['path','action','bytes','sha256','old_sha256','old_lf_sha256'],delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    manifest=b.getvalue().encode()
    out.mkdir(parents=True)
    for rel,content in data.items():
        dest=out/'payload'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    (out/'UPDATE_MANIFEST.tsv').write_bytes(manifest)
    (out/'UPDATE_METADATA.json').write_text(json.dumps(dict(update_id=a.id,repository_url=git('remote','get-url','origin'),base_commit=base,file_count=len(rows),manifest_sha256=sha(manifest)),indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('Update-HGV.ps1'),out/'Update-HGV.ps1')
    for name,mode in [('01_CHECK','Plan'),('02_APPLY','Apply'),('03_COMMIT_AND_PUSH','Publish'),('04_VERIFY','Verify')]:
        (out/(name+'.cmd')).write_bytes(('@echo off\r\nsetlocal\r\nset "PSModulePath="\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Update-HGV.ps1" -Mode '+mode+' -RepoPath "%~1"\r\nset "HGV_EXIT=%ERRORLEVEL%"\r\nif "%~1"=="" pause\r\nexit /b %HGV_EXIT%\r\n').encode())
    helper=Path(__file__).with_name('OneClick-HGV.ps1')
    launcher=Path(__file__).with_name('Launch-HGV.cmd')
    if helper.is_file() and launcher.is_file():
        shutil.copyfile(helper,out/'OneClick-HGV.ps1')
        shutil.copyfile(launcher,out/'双击更新GitHub.cmd')
        (out/'LOCAL_REPOSITORY.txt').write_text(str(repo)+'\n',encoding='utf-8')
    print(f'Prepared {len(rows)} changed files in {out}. Existing identical files were omitted. No Git mutations performed.')

if __name__=='__main__':main()
