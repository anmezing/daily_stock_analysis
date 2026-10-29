import type React from 'react';
import { useAuth } from '../../hooks';
import { Badge, Button } from '../common';
import { SettingsSectionCard } from './SettingsSectionCard';

/**
 * Read-only auth status summary. Password changes are handled by
 * ``ChangePasswordCard``; this card keeps the settings page informed about
 * whether the console currently requires a login.
 */
export const AuthSettingsCard: React.FC = () => {
  const { authEnabled, passwordSet, loggedIn, refreshStatus } = useAuth();

  return (
    <SettingsSectionCard
      title="访问认证"
      description="Web 控制台的登录开关与密码状态。修改密码请使用下文的密码卡片。"
      actions={
        <Button variant="settings-secondary" size="sm" onClick={() => void refreshStatus()}>
          刷新状态
        </Button>
      }
    >
      <dl className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-white/8 bg-elevated/40 px-3 py-2">
          <dt className="text-[11px] text-muted-text">登录保护</dt>
          <dd className="mt-1">
            <Badge variant={authEnabled ? 'success' : 'warning'}>
              {authEnabled ? '已启用' : '未启用'}
            </Badge>
          </dd>
        </div>
        <div className="rounded-lg border border-white/8 bg-elevated/40 px-3 py-2">
          <dt className="text-[11px] text-muted-text">管理员密码</dt>
          <dd className="mt-1">
            <Badge variant={passwordSet ? 'success' : 'default'}>
              {passwordSet ? '已设置' : '未设置'}
            </Badge>
          </dd>
        </div>
        <div className="rounded-lg border border-white/8 bg-elevated/40 px-3 py-2">
          <dt className="text-[11px] text-muted-text">当前会话</dt>
          <dd className="mt-1">
            <Badge variant={loggedIn ? 'info' : 'default'}>
              {loggedIn ? '已登录' : '未登录'}
            </Badge>
          </dd>
        </div>
      </dl>
    </SettingsSectionCard>
  );
};
