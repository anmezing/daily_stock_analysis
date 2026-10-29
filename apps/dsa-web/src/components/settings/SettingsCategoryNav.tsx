import type React from 'react';
import { Badge } from '../common';
import { cn } from '../../utils/cn';
import {
  getCategoryDescriptionZh,
  getCategoryTitleZh,
} from '../../utils/systemConfigI18n';
import type {
  SystemConfigCategorySchema,
  SystemConfigItem,
} from '../../types/systemConfig';

interface SettingsCategoryNavProps {
  categories: SystemConfigCategorySchema[];
  itemsByCategory: Record<string, SystemConfigItem[]>;
  activeCategory: string;
  onSelect: (category: string) => void;
}

export const SettingsCategoryNav: React.FC<SettingsCategoryNavProps> = ({
  categories,
  itemsByCategory,
  activeCategory,
  onSelect,
}) => (
  <nav className="space-y-1.5" aria-label="Settings categories">
    {categories.map((entry) => {
      const active = entry.category === activeCategory;
      const count = itemsByCategory[entry.category]?.length ?? 0;
      return (
        <button
          key={entry.category}
          type="button"
          onClick={() => onSelect(entry.category)}
          className={cn(
            'w-full rounded-lg border px-3 py-2 text-left transition-colors',
            active
              ? 'border-cyan/40 bg-cyan/10 text-foreground'
              : 'border-white/8 bg-transparent text-secondary-text hover:border-white/15 hover:bg-white/5',
          )}
        >
          <span className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">
              {getCategoryTitleZh(entry.category, entry.title)}
            </span>
            {count > 0 ? <Badge variant={active ? 'info' : 'default'}>{count}</Badge> : null}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-muted-text">
            {getCategoryDescriptionZh(entry.category, entry.description || '')}
          </span>
        </button>
      );
    })}
  </nav>
);
